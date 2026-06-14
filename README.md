# 🤖 Smart Finance - AI Model Training Documentation

## Overview

Proyek ini bertujuan untuk membangun model Artificial Intelligence (AI) yang mampu memprediksi pengeluaran di bulan yang akan datang. Model dikembangkan menggunakan Python dan framework TensorFlow/Keras dengan pendekatan Deep Learning.

---

## Project Objective

Tujuan utama dari model ini adalah:

* Melakukan prediksi secara otomatis berdasarkan pola data historis.
* Membantu pengambilan keputusan berbasis data.
* Mengurangi kesalahan prediksi yang dilakukan secara manual.

---

## Dataset

Dataset yang digunakan berasal dari sumber internal proyek dan terdiri dari beberapa fitur yang relevan terhadap target prediksi.

### Data Features

| Feature   | Description                                  |
| --------- | --------------------------                   |
| total_amount | Total pengeluaran per bulan (Rupiah)      |
| jumlah_trx    | Jumlah transaksi per bulan               | 
| rata-rata    | Rata-rata nominal transaksi per bulan     |            
| std_amount    | Standar deviasi nominal transaksi        |
| month_sin    | Encoding sinus bulan (seasonality)        |
| month_cos    | Encoding cosinus bulan (seasonality)      |                 
| rolling_3    | Rata-rata pengeluaran 3 bulan sebelumnya  |
| rolling_6    | Rata-rata pengeluaran 6 bulan sebelumnya  |
| Target    | Total pengeluaran bulan depan                |

### Dataset Split

| Data           | Percentage |
| -------------- | ---------- |
| Training Set   | 85%        |
| Testing Set    | 15%        |

---

## Data Preprocessing

Tahapan preprocessing yang dilakukan sebelum model dilatih:

1. Data Extraction — Transaksi diekstrak dari PDF (digital via pdfplumber, scan via EasyOCR) atau gambar.
2. Parsing & Type Conversion — Kolom tanggal dikonversi ke datetime, nominal rupiah dikonversi ke numerik (format Indonesia: titik ribuan, koma desimal).
3. Missing Value Check — Pengecekan missing values per kolom; std_amount NaN diisi 0.
4. Monthly Aggregation — Data transaksi harian diagregasi per bulan menghasilkan 4 fitur: total_amount, jumlah_trx, rata_rata, std_amount.
5. Feature Engineering — Ditambahkan fitur siklikal (month_sin, month_cos) dan moving average (rolling_3, rolling_6).
6. Normalization — Seluruh 8 fitur dinormalisasi ke rentang [0, 1] menggunakan MinMaxScaler.
7. Sequence Creation — Data diubah ke format sliding window dengan LOOKBACK = 12 bulan sebagai input model.
8. Train-Test Split — Data dibagi secara kronologis: 85% training, 15% testing (tanpa pengacakan). 

---

## Model Architecture

Model dibangun menggunakan arsitektur Deep Learning sebagai berikut:

```text
Input (12 bulan, 8 fitur)
      ↓
Conv1D (64 filter) — filter noise
      ↓
Conv1D (32 filter) — filter noise lanjut
      ↓
Dropout
      ↓
Bidirectional LSTM (64 unit) — tangkap pola 2 arah
      ↓
Dropout
      ↓
Bidirectional LSTM (32 unit)
      ↓
Dropout
      ↓
Temporal Attention Layer — fokus ke bulan relevan
      ↓
Dense (32, relu) → Dense (16, relu)
      ↓
Output (prediksi pengeluaran bulan depan)
```

### Hyperparameters

| Parameter        | Value                          |
|------------------|--------------------------------|
| Optimizer        | Adam (lr = 0.001)              |
| Loss Function    | SmartFinanceLoss (Huber + MAE) |
| Epochs           | 100                            |
| Batch Size       | 32                             |
| Lookback Window  | 12 bulan                       |
| Train Ratio      | 0.85                           |
| Dropout          | 0.2 (CNN & LSTM), 0.15 (Dense) |
| LSTM Dropout     | 0.1                            |
| Min LR           | 1e-7                           |
| Early Stopping   | patience = 50                  |
| ReduceLROnPlateau| patience = 20, factor = 0.5    |

---

## Model Training

Model dilatih menggunakan Google Colab dengan akselerasi GPU NVIDIA Tesla T4 yang disediakan secara gratis oleh Google Colab

### Training Configuration

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=SmartFinanceLoss(delta=0.1),  # Custom: Huber + 0.5 * MAE
    metrics=['mae']
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[
        MAPELogger(X_test, y_test, scaler, target_col=0, log_every=10),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=20, factor=0.5, min_lr=1e-7)
    ],
    verbose=0
)
```

---

## Evaluation Results

Hasil evaluasi model pada testing dataset:

| Metric   | Score |
| -------- | ----- |
| MAPE      | 9.43%  |
| MAE      |  Rp 12,166,942.41  |
| RMSE     | Rp 13,955,821.54  |
| Accuracy | 90.57%  |

---

## Model Saving

Model yang telah dilatih disimpan dalam format:

```text
saved_models/
├── smart_finance.keras       # Model utama (Keras format)
├── smart_finance_savedmodel/ # SavedModel format (untuk deployment)
│   ├── saved_model.pb
│   └── variables/
├── scaler.pkl                # MinMaxScaler (untuk inverse transform)
└── metadata.json             # Konfigurasi & metrik model
```

---

## Technologies Used

- tensorflow
- scikit-learn
- pandas
- numpy
- matplotlib
- pdfplumber
- easyocr
- pdf2image
- Pillow
- mlflow

---
