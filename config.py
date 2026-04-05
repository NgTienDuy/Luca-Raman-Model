"""
Configuration for Raman Amino Acid project.
All hyperparameters, paths, and constants in one place.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── Paths ───────────────────────────────────────────────────────────────
DATA_PATH = "data.csv"
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
PREDICTIONS_DIR = "predictions"
ANALYSIS_DIR = "analysis"

# ── Laser / Conversion ─────────────────────────────────────────────────
LASER_WL_NM = 784.815734863281  # excitation wavelength

# ── Amino acid labels (column order) ───────────────────────────────────
AA_NAMES = ["Alanine", "Asparagine", "Aspartic Acid",
            "Glutamic Acid", "Histidine", "Glucosamine"]
NUM_AA = 6

# ── Data split ──────────────────────────────────────────────────────────
SEED = 42
VAL_SIZE = 6   # samples
TEST_SIZE = 6  # samples

# ── Preprocessing ───────────────────────────────────────────────────────
COSMIC_THR = 5.0
SNIP_ITER = 20
ALS_LAM = 1e5
ALS_P = 0.01
SG_WINDOW = 9
SG_POLY = 2

# ── Training (deep models) ─────────────────────────────────────────────
MAX_EPOCHS = 40
PATIENCE = 10
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# ── Augmentation limits (memory-safe) ──────────────────────────────────
N_GAUSSIAN = 2
N_MIXUP = 2
NOISE_STD = 0.02

# ── Model-specific ─────────────────────────────────────────────────────
# M1
RIDGE_ALPHAS = [1e-3, 1e-2, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
ELASTICNET_L1 = [0.1, 0.5, 0.9]

# M2
NMF_K = 8
PLSR_LV_RANGE = list(range(2, 16))

# M3
MLP_CORRECTION_HIDDEN = [64, 32]

# M4
MLP_LAYERS = [1024, 256, 128, 64]

# M5
RESNET_CHANNELS = [32, 32, 64]

# M6
PREPROC_VARIANTS = ['none', 'snv', 'full', 'sg_snv']
LAMBDA_RECON = 0.01
LAMBDA_SMOOTH = 0.001

# M7
LAMBDA_POS = 0.01
LAMBDA_INT = 0.01
LAMBDA_BOND = 0.01
LAMBDA_ATTN = 0.001

# M8
LAMBDA_CHEM = 0.01

# M9
HP_TRIALS = 6
HP_EPOCHS = 15
PRUNE_SPARSITY = 0.2

# M10
RIER_Z_DIM = 32
RIER_PCA_DIM = 30
RIER_FFT_DIM = 32
RIER_NMF_K = 8

# ── Plotting ────────────────────────────────────────────────────────────
FIG_DPI = 150
SCATTER_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']

# ── Model names ─────────────────────────────────────────────────────────
MODEL_NAMES = {
    1: "Ridge/ElasticNet + Softmax",
    2: "MCR-ALS (NMF) + PLSR",
    3: "Two-Stage Hybrid (NNLS + MLP)",
    4: "MLP (Multi-Layer Perceptron)",
    5: "1D ResNet",
    6: "Adaptive Preprocessing Optimizer",
    7: "Spectral Feature Extraction Optimizer",
    8: "Regularization & Multi-task Optimizer",
    9: "Hyperparameter & Loss Function Optimizer",
    10: "Radial Exhaustive Information Explorer",
}
