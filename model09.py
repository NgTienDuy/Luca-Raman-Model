"""
Model 09: Hyperparameter & Loss Function Optimizer.
Random/Bayesian search over loss, lr, architecture, etc.
Budget: 8-15 trials × 30 epochs, then retrain best for full epochs.
Post-training: magnitude pruning (20% sparsity).
"""
import os, time, logging, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config as C
from utils import (preprocess_batch, compute_metrics, softmax_normalize, plot_scatter_aggregated, plot_scatter_raw,
                   plot_loss_curve, save_results, set_seeds)
from model_helpers import post_process_advanced_model

log = logging.getLogger(__name__)
MODEL_NUM = 9


class FlexMLP(nn.Module):
    def __init__(self, in_dim, hidden1, hidden2, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden2, 64),
            nn.ReLU(),
            nn.Linear(64, C.NUM_AA),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


def get_loss_fn(name):
    """Return loss function by name."""
    def kl(p, t):
        p = torch.clamp(p, 1e-8, 1.0)
        t = torch.clamp(t, 1e-8, 1.0)
        return F.kl_div(p.log(), t, reduction='batchmean')

    def mse(p, t): return F.mse_loss(p, t)
    def mae(p, t): return F.l1_loss(p, t)
    def huber(p, t): return F.smooth_l1_loss(p, t)

    def js(p, t):
        p = torch.clamp(p, 1e-8, 1.0)
        t = torch.clamp(t, 1e-8, 1.0)
        m = 0.5 * (p + t)
        return 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') +
                       F.kl_div(t.log(), m, reduction='batchmean'))

    return {'kl': kl, 'mse': mse, 'mae': mae, 'huber': huber, 'js': js}[name]


def prune_model(model, sparsity=0.2):
    """Simple magnitude pruning."""
    total_pruned = 0
    total_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            with torch.no_grad():
                flat = param.abs().flatten()
                threshold = torch.quantile(flat, sparsity)
                mask = param.abs() >= threshold
                param.mul_(mask.float())
                total_pruned += (~mask).sum().item()
                total_params += param.numel()
    return total_pruned, total_params


def run(X_train, X_val, X_test, Y_train, Y_val, Y_test,
        wavenumbers=None, retrain=False, **kw):
    set_seeds()
    t0 = time.time()
    save_dir = os.path.join(C.RESULTS_DIR, f"model{MODEL_NUM:02d}")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(C.CHECKPOINTS_DIR, f"model{MODEL_NUM:02d}.pt")
    sid_test = kw.get("sid_test")

    log.info("M09: Preprocessing...")
    Xtr = preprocess_batch(X_train, 'sg_snv')
    Xv = preprocess_batch(X_val, 'sg_snv')
    Xt = preprocess_batch(X_test, 'sg_snv')

    D = Xtr.shape[1]
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    Ytr_t = torch.tensor(Y_train, dtype=torch.float32)
    Yv_t = torch.tensor(Y_val, dtype=torch.float32)

    # Define search space
    hp_space = []
    rng = np.random.RandomState(C.SEED)
    for _ in range(C.HP_TRIALS):
        hp_space.append({
            'loss': rng.choice(['kl', 'mse', 'mae', 'huber', 'js']),
            'lr': float(10 ** rng.uniform(-4, -2)),
            'wd': float(10 ** rng.uniform(-5, -2)),
            'drop': float(rng.uniform(0.1, 0.5)),
            'h1': int(rng.choice([128, 192, 256, 320])),
            'h2': int(rng.choice([64, 96, 128, 192])),
            'batch_size': int(rng.choice([64, 128])),
        })

    log.info(f"M09: Running {len(hp_space)} HP trials ({C.HP_EPOCHS} epochs each)...")
    trial_results = []

    for trial_idx, hp in enumerate(hp_space):
        set_seeds(C.SEED + trial_idx)
        model = FlexMLP(D, hp['h1'], hp['h2'], hp['drop'])
        opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
        loss_fn = get_loss_fn(hp['loss'])

        ds = TensorDataset(Xtr_t, Ytr_t)
        loader = DataLoader(ds, batch_size=hp['batch_size'], shuffle=True)

        best_vl = float('inf')
        for epoch in range(C.HP_EPOCHS):
            model.train()
            for xb, yb in loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                vp = model(Xv_t)
                # Always use KL for evaluation consistency
                vl = F.kl_div(torch.clamp(vp, 1e-8).log(),
                              torch.clamp(Yv_t, 1e-8), reduction='batchmean').item()
            best_vl = min(best_vl, vl)

        trial_results.append({'hp': hp, 'val_loss': best_vl, 'trial': trial_idx})
        log.info(f"  Trial {trial_idx+1}/{len(hp_space)}: loss={hp['loss']}, "
                 f"lr={hp['lr']:.5f}, val={best_vl:.4f}")

        del model, opt
        gc.collect()

    # Find best
    best_trial = min(trial_results, key=lambda x: x['val_loss'])
    best_hp = best_trial['hp']
    log.info(f"M09: Best HP: {best_hp}")

    # Full retrain with best HP
    set_seeds()
    model = FlexMLP(D, best_hp['h1'], best_hp['h2'], best_hp['drop'])
    opt = torch.optim.AdamW(model.parameters(), lr=best_hp['lr'], weight_decay=best_hp['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    loss_fn = get_loss_fn(best_hp['loss'])

    ds = TensorDataset(Xtr_t, Ytr_t)
    loader = DataLoader(ds, batch_size=best_hp['batch_size'], shuffle=True)

    train_losses, val_losses = [], []
    best_val = float('inf')
    patience_ctr = 0

    log.info("M09: Full retrain with best HP...")
    for epoch in range(C.MAX_EPOCHS):
        model.train()
        ep_loss = 0
        nb = 0
        for xb, yb in loader:
            # Augmentation: noise
            xb_n = xb + torch.randn_like(xb) * C.NOISE_STD
            pred = model(xb_n)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1

        sched.step()
        train_losses.append(ep_loss / max(nb, 1))

        model.eval()
        with torch.no_grad():
            vp = model(Xv_t)
            vl = F.kl_div(torch.clamp(vp, 1e-8).log(),
                          torch.clamp(Yv_t, 1e-8), reduction='batchmean').item()
        val_losses.append(vl)

        if vl < best_val:
            best_val = vl
            patience_ctr = 0
            torch.save({'model': model.state_dict(), 'hp': best_hp, 'epoch': epoch + 1}, ckpt_path)
        else:
            patience_ctr += 1

        if (epoch + 1) % 10 == 0:
            log.info(f"  Epoch {epoch+1}: train={train_losses[-1]:.4f}, val={vl:.4f}")
        if patience_ctr >= C.PATIENCE:
            log.info(f"  Early stopping at epoch {epoch+1}")
            break

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False)['model'])

    # Pruning
    log.info(f"M09: Pruning with {C.PRUNE_SPARSITY*100}% sparsity...")
    pruned, total = prune_model(model, C.PRUNE_SPARSITY)
    log.info(f"  Pruned {pruned}/{total} parameters ({100*pruned/max(total,1):.1f}%)")

    model.eval()
    with torch.no_grad():
        Y_pred = model(Xt_t).numpy()
    Y_pred = softmax_normalize(Y_pred)

    metrics = compute_metrics(Y_test, Y_pred, D)
    metrics["epochs"] = len(train_losses)
    metrics["training_time"] = time.time() - t0
    metrics["best_hp"] = {k: str(v) for k, v in best_hp.items()}
    metrics["trial_results"] = [{"trial": r['trial'], "val_loss": r['val_loss'],
                                  "loss_fn": r['hp']['loss']} for r in trial_results]
    metrics["pruned_params"] = pruned
    metrics["total_params"] = total
    metrics["sparsity"] = pruned / max(total, 1)

    log.info(f"M09: Test R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")

    plot_loss_curve(train_losses, val_losses, MODEL_NUM, save_dir)
    plot_scatter_aggregated(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    plot_scatter_raw(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    save_results(metrics, MODEL_NUM, save_dir)

    avg_chem = post_process_advanced_model(Xt, wavenumbers, Y_pred, sid_test, MODEL_NUM, save_dir)
    metrics["chemistry"] = avg_chem

    gc.collect()
    return Y_pred, metrics
