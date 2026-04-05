"""
Model 05: 1D ResNet with skip connections.
Strided convolutions for CPU efficiency. Proper training schedule.
"""
import os, time, logging, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config as C
from utils import (preprocess_batch, compute_metrics, softmax_normalize,
                   plot_scatter_aggregated, plot_scatter_raw,
                   plot_loss_curve, save_results, set_seeds)

log = logging.getLogger(__name__)
MODEL_NUM = 5


class ResBlock1D(nn.Module):
    def __init__(self, ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ResNet1D(nn.Module):
    def __init__(self, in_len=1024, drop=0.3):
        super().__init__()
        # Stem with stride to reduce length fast: 1024 → 256 → 64
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=4, padding=3),   # → 256
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.block1 = ResBlock1D(32, 5)
        self.down1 = nn.Sequential(
            nn.Conv1d(32, 64, 3, stride=4, padding=1),  # → 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.block2 = ResBlock1D(64, 3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(64, C.NUM_AA)

    def forward(self, x):
        x = x.unsqueeze(1)       # (B, 1, L)
        x = self.stem(x)         # (B, 32, L/4)
        x = self.block1(x)
        x = self.down1(x)        # (B, 64, L/16)
        x = self.block2(x)
        x = self.gap(x).squeeze(-1)  # (B, 64)
        x = self.drop(x)
        return F.softmax(self.fc(x), dim=-1)


def kl_loss(pred, target):
    pred = torch.clamp(pred, 1e-8, 1.0)
    target = torch.clamp(target, 1e-8, 1.0)
    return F.kl_div(pred.log(), target, reduction='batchmean')


def run(X_train, X_val, X_test, Y_train, Y_val, Y_test,
        wavenumbers=None, retrain=False, **kw):
    set_seeds()
    t0 = time.time()
    save_dir = os.path.join(C.RESULTS_DIR, f"model{MODEL_NUM:02d}")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(C.CHECKPOINTS_DIR, f"model{MODEL_NUM:02d}.pt")
    sid_test = kw.get("sid_test")

    log.info("M05: Preprocessing...")
    Xtr = preprocess_batch(X_train, 'sg_snv')
    Xv = preprocess_batch(X_val, 'sg_snv')
    Xt = preprocess_batch(X_test, 'sg_snv')
    in_dim = Xtr.shape[1]

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); del Xtr
    Ytr_t = torch.tensor(Y_train, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32); del Xv
    Yv_t = torch.tensor(Y_val, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32); del Xt
    gc.collect()

    model = ResNet1D(in_len=in_dim, drop=C.DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    start_epoch = 0
    if os.path.exists(ckpt_path) and not retrain:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    train_ds = TensorDataset(Xtr_t, Ytr_t)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    log.info(f"M05: Training from epoch {start_epoch}...")
    for epoch in range(start_epoch, C.MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        n_b = 0

        for xb, yb in train_loader:
            # In-batch noise augmentation
            xb = xb + torch.randn_like(xb) * C.NOISE_STD
            pred = model(xb)
            loss = kl_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        scheduler.step()
        train_losses.append(epoch_loss / max(n_b, 1))

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_t)
            vl = kl_loss(val_pred, Yv_t).item()
        val_losses.append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch + 1}, ckpt_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            log.info(f"  Epoch {epoch+1}: train={train_losses[-1]:.4f}, val={vl:.4f}")

        if patience_counter >= C.PATIENCE:
            log.info(f"  Early stopping at epoch {epoch+1}")
            break

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False)['model'])

    model.eval()
    with torch.no_grad():
        Y_pred = model(Xt_t).numpy()
    Y_pred = softmax_normalize(Y_pred)

    metrics = compute_metrics(Y_test, Y_pred, in_dim)
    metrics["epochs"] = len(train_losses)
    metrics["training_time"] = time.time() - t0

    log.info(f"M05: Test R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")

    plot_loss_curve(train_losses, val_losses, MODEL_NUM, save_dir)
    plot_scatter_aggregated(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    plot_scatter_raw(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    save_results(metrics, MODEL_NUM, save_dir)

    gc.collect()
    return Y_pred, metrics
