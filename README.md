---
title: Rossmann Sales Prediction
emoji: 🏪
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
---

# 🏪 Rossmann Store Sales Prediction

**Nama:** Rizki Hasan Fauzi  
**NIM:** 301240071  

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

> Proyek Machine Learning untuk memprediksi penjualan harian toko retail Rossmann menggunakan **5 algoritma ML/AI**, dilengkapi dengan web dashboard interaktif.

---

## 🔗 Tautan Penting

- **Demo Aplikasi:** [Klik di sini untuk Demo](https://...)
- **Laporan Proyek:** [Klik di sini untuk Laporan](https://...)
- **Video Presentasi:** [Klik di sini untuk Video YouTube](https://...)

---

## 📊 Dataset

**Sumber:** [Kaggle — Rossmann Store Sales](https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales/data)  
**Lisensi:** Database Contents License (DbCL) / Kaggle Competition Rules

| File | Deskripsi | Ukuran |
|------|-----------|--------|
| `train.csv` | Data training (1,017,209 baris) | ~38 MB |
| `test.csv` | Data testing | ~1.4 MB |
| `store.csv` | Informasi 1,115 toko | ~45 KB |

> **Note:** File `train.csv` dan `test.csv` tidak disertakan di repository karena ukurannya. Silakan unduh dari [Kaggle](https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales/data) dan letakkan di folder `data/`.

---

## 🤖 5 Algoritma yang Digunakan

| # | Model | Kategori | Deskripsi |
|---|-------|----------|-----------|
| 1 | **Linear Regression** | Supervised — Regression | Model baseline untuk prediksi |
| 2 | **ANN** | Supervised — Deep Learning | Artificial Neural Network dengan 3 hidden layers |
| 3 | **LSTM** | Supervised — Deep Learning | Recurrent Neural Network untuk time series |
| 4 | **K-Means Clustering** | Unsupervised — Clustering | Segmentasi toko berdasarkan fitur |
| 5 | **Backpropagation** | Supervised — Deep Learning | Custom NN dengan explicit gradient computation |

---

## 📁 Struktur Project

```
rossman_project/
│
├── 📂 data/                           ← Dataset CSV
│   ├── train.csv                        (download dari Kaggle)
│   ├── test.csv                         (download dari Kaggle)
│   └── store.csv
│
├── 📂 notebooks/                      ← Jupyter Notebooks
│   ├── 01_preprocessing_eda.ipynb       ← Preprocessing & EDA
│   └── 02_model_implementation.ipynb    ← Training 5 Algoritma ML
│
├── 📂 models/                         ← Trained models (auto-generated)
│   ├── linear_regression.pkl
│   ├── ann_model.h5
│   ├── lstm_model.h5
│   ├── kmeans_model.pkl
│   └── backprop_model.h5
│
├── 📂 app/                            ← Flask Web Application
│   ├── app.py                           ← Main Flask app
│   ├── __init__.py
│   ├── static/
│   │   ├── style.css                    ← Custom styling
│   │   ├── logo_rossmann.png
│   │   └── hero_bg.png
│   └── templates/
│       ├── home.html                    ← Dashboard
│       ├── predict.html                 ← Prediksi penjualan
│       ├── comparison.html              ← Perbandingan model
│       ├── info.html                    ← Info dataset & arsitektur
│       ├── 404.html
│       └── 500.html
│
├── 📂 docs/                           ← Visualisasi & hasil (auto-generated)
│   ├── 01_preprocessing_eda/            ← Output EDA
│   └── 02_model_evaluation/            ← Output evaluasi model
│
├── .gitignore
├── Procfile                            ← Deployment config (Heroku/Railway)
├── requirements.txt
└── README.md
```

---

## 🚀 Cara Menjalankan

### Prerequisites

- Python 3.10+
- pip

### 1. Clone Repository

```bash
git clone https://github.com/rizkihsn/rossman_project.git
cd rossman_project
```

### 2. Buat Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download dataset dari [Kaggle](https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales/data), lalu letakkan file `train.csv` dan `test.csv` ke dalam folder `data/`.

### 5. Jalankan Notebooks

Buka dan jalankan notebook secara berurutan:

```bash
jupyter notebook
```

1. **`01_preprocessing_eda.ipynb`** — Preprocessing & Exploratory Data Analysis
   - Load & merge dataset (train + store, test + store)
   - Handle missing values & feature engineering
   - Membuat visualisasi EDA di `docs/01_preprocessing_eda/`
   - Menyimpan preprocessed data ke `models/`

2. **`02_model_implementation.ipynb`** — Training 5 Algoritma ML
   - Training & evaluasi semua model
   - Evaluasi dengan MAE, MSE, RMSE, R²
   - Menyimpan model ke `models/`
   - Membuat visualisasi hasil di `docs/02_model_evaluation/`

### 6. Jalankan Web App

```bash
python app/app.py
```

Buka browser: **http://localhost:5000**

---

## 🌐 Fitur Web App

| Halaman | URL | Fungsi |
|---------|-----|--------|
| 🏠 Home | `/` | Dashboard & info project |
| 🔮 Predict | `/predict` | Input data → prediksi sales dengan 4 model |
| 📊 Comparison | `/comparison` | Perbandingan performa semua model |
| ℹ️ Info | `/info` | Detail dataset & arsitektur model |

---

## 📈 Contoh Output Visualisasi

<details>
<summary><b>📊 EDA — Exploratory Data Analysis</b></summary>

Visualisasi yang dihasilkan dari notebook EDA:
- Distribusi penjualan
- Correlation matrix antar fitur

</details>

<details>
<summary><b>🤖 Model Evaluation</b></summary>

Setiap model menghasilkan 3 jenis visualisasi:
- **Loss / Convergence Chart** — Grafik konvergensi training
- **Actual vs Predicted** — Perbandingan nilai aktual & prediksi
- **Error Distribution** — Distribusi error prediksi

</details>

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
|----------|-----------|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **Deep Learning** | TensorFlow / Keras |
| **Web Framework** | Flask |
| **Frontend** | Bootstrap 5, Chart.js |
| **Visualization** | Matplotlib, Seaborn |

---

## 📝 Lisensi

Project ini dibuat untuk keperluan akademik.  
Dataset menggunakan lisensi dari [Kaggle](https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales/data).

---

<p align="center">
  <i>Built with ❤️ using Python, Flask & TensorFlow</i>
</p>
