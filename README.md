# Phân tích Phổ Raman bằng Học Máy — v4
### 10 Mô hình + Phân tích Hóa-Lý Toàn diện

---

## Cấu trúc Dự án

```
raman_aa/
├── config.py              # Cấu hình, hyperparameter
├── utils.py               # Tiện ích: load, preprocess, eval, checkpoint
├── spectral_knowledge.py  # Tri thức hóa học bond regions (267–2004 cm⁻¹)
├── chemistry_report.py    # Trích xuất & báo cáo thông tin hóa-lý  ← MỚI
│
├── model01.py   Softmax Regression
├── model02.py   PCA + Ridge Regression
├── model03.py   MCR-ALS (NMF) + PLSR
├── model04.py   ANN (MLP) + augmentation
├── model05.py   1D CNN + augmentation
├── model06.py   Bond Analysis + XGBoost + chemistry
├── model07.py   CNN + Bond-Region Attention + chemistry
├── model08.py   Neural Module Network + chemistry
├── model09.py   RL Pipeline Discovery + chemistry
├── model10.py   RIER (Radial Information Expansion) + full chemistry
│
├── run_all_models.py      # File chạy chính
└── README.md
```

---

## Cài đặt

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch xgboost
```

---

## Định dạng Dữ liệu

File CSV 4379 × 1031:
- **1024 cột đầu**: wavelength từ 801.62→931.28 nm *(tự động chuyển sang Raman shift 267–2004 cm⁻¹)*
- **Cột `vial #`**: tên sample (a01–a48, DL-alanine, L-histidine, ...)
- **6 cột cuối**: Alanine, Asparagine, Aspartic Acid, Glutamic Acid, Histidine, Glucosamine

---

## Cách Chạy

```bash
# Tất cả 10 model
python run_all_models.py --data data.csv

# Model cụ thể
python run_all_models.py --data data.csv --models 3,6,10

# Train lại từ đầu
python run_all_models.py --data data.csv --retrain

# Chỉ phân tích hóa-lý (cần đã có checkpoint)
python run_all_models.py --data data.csv --chemistry-only

# Chỉ xem bảng so sánh
python run_all_models.py --compare-only
```

---

## Pipeline Xử lý Dữ liệu

```
data.csv (4378 phổ, 54 samples)
    ↓  wavelength → Raman shift (267–2004 cm⁻¹)
    ↓  aggregate: median 90 phổ/sample → 1 phổ đại diện
    ↓  split: 48 train samples / 6 test samples
    ↓  preprocessing: cosmic ray → ALS baseline → SG smooth → SNV
    ↓  10 models train/predict
    ↓  scatter: predict trên TẤT CẢ raw spectra (6 nhóm × 90 điểm)
    ↓  chemistry: 15+ tính chất hóa-lý từ models 6–10
```

---

## Thông tin Hóa-Lý Trích Xuất (Models 6–10)

| Tính chất | Công thức / Nguồn |
|-----------|-------------------|
| pH xu hướng | COO⁻ vs NH₃⁺/NH₂ balance |
| Độ phân cực | weighted bond polarity |
| Tính ưa nước | weighted hydrophilicity |
| Bền liên kết | weighted bond strength |
| Tính thơm | C=C/C=N region (1560–1640 cm⁻¹) |
| Amide I/III ratio | C=O / N-H·C-H (cấu trúc bậc 2) |
| Cấu trúc bậc 2 | Amide I vị trí → α-helix/β-sheet |
| Aliphatic index | CH₂/CH₃ (1430–1480 cm⁻¹) |
| Carboxylate ratio | COO⁻ (1380–1430 cm⁻¹) |
| Amine ratio | NH₃⁺/NH₂ (1480–1560 cm⁻¹) |
| C-N/C-C ratio | Amine khung / carbon backbone |
| Crystallinity proxy | 1/mean(FWHM) |
| Spectral entropy | Shannon entropy của phổ |
| Estimated pI | từ tỉ lệ AA + tham chiếu |
| Estimated MW | từ tỉ lệ AA + tham chiếu |

### Model 10 (RIER) có thêm:
- Spoke × Chemistry MI (mutual information của mỗi spoke với tính chất hóa học)
- Per-sample physicochemical table
- Radar chart so sánh 8 spokes

---

## File Đầu Ra

```
results/
├── comparison.png                    # R² và MAE so sánh
├── comparison_table.csv
├── model0X_scatter.png              # Scatter 6 sample (aggregated)
├── model0X_scatter_raw.png          # Scatter raw spectra (6 nhóm điểm) ← MỚI
├── model0X_loss.png
├── model06_feature_importance.png
├── model06_chemistry_bonds.png      # Bond contribution M6
├── model07_bond_attention.png
├── model07_chemistry_bonds.png
├── model08_chemistry_bonds.png
├── model09_chemistry_bonds.png
├── model10_chemistry_bonds.png
├── model10_radial_info.png
├── chemistry_comparison_radar.png   # Radar so sánh M6-M10 ← MỚI
└── chemistry_bond_heatmap.png       # Heatmap bond × model ← MỚI

analysis/
├── chemistry_profiles.json         # Tất cả profiles JSON ← MỚI
├── per_sample_chemistry.json        # Per-sample hóa-lý ← MỚI
├── model06_chemistry.json
├── model07_chemistry.json
├── model08_chemistry.json
├── model09_chemistry.json
└── model10_chemistry.json           # RIER full analysis ← MỚI
```

---

## Chiến lược Chống Overfit (N_train=48)

| Kỹ thuật | Models |
|----------|--------|
| Augmentation (Gaussian noise + Mixup) | M4, M5, M7, M8, M9, M10 |
| LOO Cross-validation | M3 (PLSR n_lv) |
| Dropout 0.5 | M4, M5, M7 |
| Weight decay 1e-3 | M4–M10 |
| L1+L2 regularization | M6 (XGBoost) |
| Ridge α=100 | M1 |
| Diversity loss | M10 (RIER) |

---

## Kết Quả Đã Ghi Nhận (seed=42)

| # | Mô hình | R² Test | MAE |
|---|---------|---------|-----|
| 3 | MCR-ALS + PLSR | **0.831** | 0.025 |
| 10 | RIER | **0.915** | 0.016 |
| 6 | Bond + XGBoost | 0.482 | 0.044 |

---

## Cơ Chế Resume

```
Ctrl+C lần 1 → dừng sau epoch hiện tại, lưu checkpoint
Ctrl+C lần 2 → dừng ngay
Hết RAM      → bỏ qua model đó, tiếp tục
Tắt máy      → chạy lại, tự tải resume checkpoint
```

---

## Wavelength → Raman Shift

```
Laser: 784.815734863281 nm
Shift (cm⁻¹) = ((1/784.815734863281) - (1/λ_nm)) × 10⁷

801.62 nm → 267.1 cm⁻¹  (C-C-C skeletal)
850.00 nm → 977.1 cm⁻¹  (C-N amine)
900.00 nm → 1630.7 cm⁻¹ (C=O amide I)
931.28 nm → 2003.9 cm⁻¹ (C-H stretch)
```

*Tự động phát hiện và chuyển đổi khi đọc CSV.*
