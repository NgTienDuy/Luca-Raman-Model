"""config.py — Cấu hình toàn dự án"""
import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
ANALYSIS_DIR    = os.path.join(BASE_DIR, "analysis")
COMPARISON_DIR  = os.path.join(RESULTS_DIR, "comparison")

def get_model_dir(model_id: int) -> str:
    """Trả về thư mục kết quả của từng model"""
    d = os.path.join(RESULTS_DIR, f"model{model_id:02d}")
    os.makedirs(d, exist_ok=True)
    return d

for d in [RESULTS_DIR, CHECKPOINT_DIR, PREDICTIONS_DIR, ANALYSIS_DIR, COMPARISON_DIR]:
    os.makedirs(d, exist_ok=True)

DATA_FILE   = "data.csv"
LABEL_COLS  = ["Alanine","Asparagine","Aspartic Acid",
               "Glutamic Acid","Histidine","Glucosamine"]
SAMPLE_COL  = "vial #"
N_OUTPUTS   = 6
LASER_NM    = 784.815734863281

TRAIN_SAMPLES       = 48
SPLIT_SEED          = 2026
SAVE_EVERY_N_EPOCHS = 10

MODEL_CONFIGS = {
    1:  {"alpha":100,"solver":"lsqr","max_iter":5000},
    2:  {"n_components":40,"alpha":0.1,"pca_variance":0.99},
    3:  {"n_components_nmf":8,"n_lv_plsr":20,"max_iter_nmf":3000,"n_bond_regions":8},
    4:  {"hidden_sizes":[256,128,64],"dropout":0.5,"lr":5e-4,"weight_decay":1e-3,
         "batch_size":16,"max_epochs":500,"patience":50,"n_aug":10,"noise_std":0.02},
    5:  {"channels":[32,64,128],"kernels":[15,9,5],"dropout":0.5,"lr":5e-4,
         "weight_decay":1e-3,"batch_size":16,"max_epochs":500,"patience":50,"n_aug":8},
    6:  {"n_estimators":600,"learning_rate":0.03,"max_depth":4,
         "early_stopping_rounds":30,"subsample":0.8,"colsample_bytree":0.8,
         "lambda":5.0,"alpha":2.0},
    7:  {"channels":[32,64,128],"kernels":[15,9,5],"dropout":0.5,"entropy_lambda":5e-4,
         "lr":5e-4,"weight_decay":1e-3,"batch_size":16,"max_epochs":500,"patience":50,"n_aug":8},
    8:  {"n_filters":32,"n_bond_regions":13,"recon_weight":0.05,"chem_weight":0.01,
         "lr":5e-4,"weight_decay":1e-3,"batch_size":16,"max_epochs":400,"patience":50,"n_aug":8},
    9:  {"n_tools":8,"max_steps":4,"gru_hidden":64,"policy_weight":0.05,"recon_weight":0.05,
         "lr":5e-4,"weight_decay":1e-3,"batch_size":16,"max_epochs":300,"patience":50,"n_aug":6},
    10: {"latent_dim":64,"n_fft_components":64,"n_nmf_components":8,"recon_weight":0.03,
         "diversity_weight":0.01,"lr":5e-4,"weight_decay":1e-3,"batch_size":16,
         "max_epochs":400,"patience":50,"n_aug":8},
}

MODEL_NAMES = {
    1:"Softmax Regression", 2:"PCA + Ridge Regression",
    3:"MCR-ALS (NMF) + PLSR", 4:"ANN (MLP)", 5:"1D CNN",
    6:"Spectral Bond + XGBoost", 7:"CNN + Bond-Region Attention",
    8:"Neural Module Network", 9:"RL Pipeline Discovery",
    10:"Radial Information Expansion",
}
