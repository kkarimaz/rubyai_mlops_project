# Wine Recognition MLOps Project

Proyek ini adalah implementasi **end-to-end MLOps** untuk klasifikasi jenis anggur menggunakan dataset UCI Wine. Proyek ini mendemonstrasikan siklus hidup Machine Learning yang lengkap — mulai dari versioning data, experiment tracking, hingga deployment menggunakan Docker dan Hugging Face Spaces.

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Tech Stack](#tech-stack)
- [Struktur Folder](#struktur-folder)
- [Cara Menjalankan](#cara-menjalankan)
  - [1. Setup Environment](#1-setup-environment)
  - [2. Siapkan Data dengan DVC](#2-siapkan-data-dengan-dvc)
  - [3. Training & Experiment Tracking](#3-training--experiment-tracking)
  - [4. Jalankan API Lokal](#4-jalankan-api-lokal)
  - [5. Jalankan dengan Docker](#5-jalankan-dengan-docker)
- [Dokumentasi API](#dokumentasi-api)
  - [GET /](#get-)
  - [GET /health](#get-health)
  - [POST /predict](#post-predict)
- [Dataset & Model](#dataset--model)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [DVC Data Versioning](#dvc-data-versioning)

---

## Gambaran Umum

Proyek ini mengklasifikasikan sampel anggur ke dalam 3 jenis kultivar Italia berdasarkan 13 fitur kimiawi:

| Label ID | Nama Kultivar |
|----------|---------------|
| 0        | Barolo        |
| 1        | Grignolino    |
| 2        | Barbera       |

**MLOps Capabilities yang diimplementasikan:**

- **Data Versioning** — DVC + DagsHub Storage memastikan dataset dapat direproduksi di environment mana pun.
- **Experiment Tracking** — MLflow secara otomatis mencatat hyperparameter, metrik (Accuracy & F1-Score), dan artifact model untuk setiap run.
- **Model Registry** — Model terbaik disimpan sebagai file `.pkl` secara lokal dan di-log ke MLflow Artifacts.
- **Production API** — FastAPI dengan validasi input menggunakan Pydantic V2 untuk melayani prediksi.
- **Containerization** — Docker image siap-deploy ke server atau platform cloud seperti Hugging Face Spaces.

---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│  UCI Wine Dataset → prepare_data.py → data/wine.csv         │
│                         ↓                                   │
│              DVC Track → DagsHub Storage                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      TRAINING LAYER                         │
│  train.py (3 hyperparameter combinations)                   │
│       ↓ logs params, metrics, model artifacts               │
│  MLflow Tracking Server (hosted on DagsHub)                 │
│       ↓ best model saved                                    │
│  models/model.pkl                                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      SERVING LAYER                          │
│  FastAPI (api/main.py)                                      │
│  POST /predict → RandomForestClassifier → Wine Label        │
│       ↓                                                     │
│  Docker Container (port 7860)                               │
│  → Deployable to Hugging Face Spaces / any server           │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Komponen            | Teknologi              | Versi      |
|---------------------|------------------------|------------|
| ML Framework        | Scikit-Learn           | 1.8.0      |
| API Framework       | FastAPI + Uvicorn      | 0.136.1    |
| Data Validation     | Pydantic               | 2.13.0     |
| Experiment Tracking | MLflow                 | 3.11.1     |
| Data Versioning     | DVC                    | 3.67.1     |
| MLOps Platform      | DagsHub                | -          |
| Model Serialization | Joblib                 | 1.5.3      |
| Data Processing     | Pandas + NumPy         | 2.3.3 / 2.4.4 |
| Containerization    | Docker (python:3.12-slim) | -       |

---

## Struktur Folder

```
rubyai_mlops_project/
│
├── api/
│   └── main.py             # FastAPI app: endpoint /, /health, /predict
│
├── data/
│   ├── wine.csv            # Dataset lokal (di-ignore oleh Git, ditrack DVC)
│   └── wine.csv.dvc        # DVC pointer file (di-track Git)
│
├── models/
│   └── model.pkl           # Model terbaik hasil training (di-ignore Git)
│
├── src/
│   ├── prepare_data.py     # Mengambil dataset dari scikit-learn, simpan ke CSV
│   └── train.py            # Training pipeline + MLflow logging
│
├── .dvc/
│   └── config              # Konfigurasi DVC remote (DagsHub)
│
├── Dockerfile              # Build image untuk deployment (port 7860)
├── requirements.txt        # Dependencies development (termasuk DVC, MLflow)
├── requirements-prod.txt   # Dependencies production (tanpa DVC, MLflow)
└── README.md
```

---

## Cara Menjalankan

### 1. Setup Environment

**Clone repository dan install dependencies:**

```bash
git clone https://dagshub.com/kkarimaz/rubyai_mlops_project.git
cd rubyai_mlops_project

python -m venv venv
source venv/bin/activate        # Linux/Mac
# atau: venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

### 2. Siapkan Data dengan DVC

> **Opsi A — Pull dari DagsHub (direkomendasikan):**
>
> Pastikan kamu sudah punya akses ke DagsHub remote. Jalankan:
>
> ```bash
> dvc pull
> ```
>
> Ini akan mengunduh `data/wine.csv` dari DagsHub Storage.

> **Opsi B — Generate ulang dari source:**
>
> ```bash
> python src/prepare_data.py
> ```
>
> Script ini mengambil dataset langsung dari `sklearn.datasets.load_wine` dan menyimpannya ke `data/wine.csv`.

---

### 3. Training & Experiment Tracking

```bash
python src/train.py
```

Script ini akan:
1. Membaca `data/wine.csv`
2. Melatih **3 variasi** `RandomForestClassifier` dengan kombinasi hyperparameter berbeda:

   | Variasi | `n_estimators` | `max_depth` | Deskripsi     |
   |---------|----------------|-------------|---------------|
   | 1       | 10             | 3           | Model ringan  |
   | 2       | 50             | 5           | Model medium  |
   | 3       | 100            | 10          | Model kompleks|

3. Mencatat hyperparameter, Accuracy, dan F1-Score ke **MLflow** (hosted di DagsHub).
4. Menyimpan model terbaik (berdasarkan F1-Score) ke `models/model.pkl`.

**Lihat hasil eksperimen di:**
`https://dagshub.com/kkarimaz/rubyai_mlops_project`

---

### 4. Jalankan API Lokal

> Pastikan `models/model.pkl` sudah ada (jalankan training terlebih dahulu).

```bash
uvicorn api.main:app --reload --port 8000
```

API akan berjalan di `http://localhost:8000`.

Buka dokumentasi interaktif (Swagger UI) di:
`http://localhost:8000/docs`

---

### 5. Jalankan dengan Docker

**Build image:**

```bash
docker build -t wine-mlops .
```

**Jalankan container:**

```bash
docker run -p 7860:7860 wine-mlops
```

API akan berjalan di `http://localhost:7860`.

> Image ini menggunakan `requirements-prod.txt` yang hanya berisi library yang dibutuhkan untuk serving (FastAPI, Scikit-Learn, Joblib, dll) — tanpa MLflow dan DVC — agar ukuran image lebih kecil.

---

## Dokumentasi API

### GET /

Health check sederhana.

**Response:**
```json
{
  "message": "Hello, World!"
}
```

---

### GET /health

Cek status layanan.

**Response:**
```json
{
  "status": "healthy"
}
```

---

### POST /predict

Menerima 13 fitur kimiawi anggur dan mengembalikan prediksi kelas kultivar.

**Request Body (JSON):**

| Field                          | Tipe    | Deskripsi                                  |
|--------------------------------|---------|--------------------------------------------|
| `alcohol`                      | float   | Kadar alkohol                              |
| `malic_acid`                   | float   | Kadar asam malat                           |
| `ash`                          | float   | Kadar abu                                  |
| `alcalinity_of_ash`            | float   | Alkalinitas abu                            |
| `magnesium`                    | float   | Kadar magnesium                            |
| `total_phenols`                | float   | Total fenol                                |
| `flavanoids`                   | float   | Kadar flavanoid                            |
| `nonflavanoid_phenols`         | float   | Kadar fenol non-flavanoid                  |
| `proanthocyanins`              | float   | Kadar proantosianin                        |
| `color_intensity`              | float   | Intensitas warna                           |
| `hue`                          | float   | Hue (corak warna)                          |
| `od280_od315_of_diluted_wines` | float   | Rasio OD280/OD315 pada anggur yang diencerkan |
| `proline`                      | float   | Kadar prolin                               |

**Contoh Request:**
```json
{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100.0,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26,
  "proanthocyanins": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.4,
  "proline": 1050.0
}
```

**Contoh Response:**
```json
{
  "message": "Prediction created!",
  "predicted_class_id": 0,
  "predicted_class_label": "Barolo"
}
```

**Contoh dengan `curl`:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14,
       "alcalinity_of_ash": 11.2, "magnesium": 100.0,
       "total_phenols": 2.65, "flavanoids": 2.76,
       "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28,
       "color_intensity": 4.38, "hue": 1.05,
       "od280_od315_of_diluted_wines": 3.4, "proline": 1050.0
     }'
```

---

## Dataset & Model

- **Dataset:** [UCI Wine Dataset](https://archive.ics.uci.edu/dataset/109/wine) via `sklearn.datasets.load_wine`
- **Jumlah sampel:** 178 (split 80/20 untuk train/test)
- **Jumlah fitur:** 13 fitur kimiawi
- **Jumlah kelas:** 3 kultivar anggur
- **Algoritma:** `RandomForestClassifier` (Scikit-Learn)
- **Metrik evaluasi:** Accuracy & F1-Score (weighted)

---

## MLflow Experiment Tracking

Semua run tercatat di MLflow Tracking Server yang dihosting di DagsHub:

```
https://dagshub.com/kkarimaz/rubyai_mlops_project.mlflow
```

Nama experiment: `Wine_Classification_RF`

Setiap run mencatat:
- **Parameters:** `n_estimators`, `max_depth`
- **Metrics:** `accuracy`, `f1_score`
- **Artifacts:** file model `.pkl`

---

## DVC Data Versioning

Dataset dikelola menggunakan DVC dengan remote storage di DagsHub:

```
https://dagshub.com/kkarimaz/rubyai_mlops_project.dvc
```

File `data/wine.csv.dvc` di-commit ke Git sebagai pointer ke versi data yang tepat. File data aktual (`wine.csv`) tidak di-push ke Git, melainkan ke DagsHub Storage.

**Perintah DVC yang umum digunakan:**

```bash
dvc pull          # Unduh data dari remote
dvc push          # Upload data ke remote
dvc status        # Cek apakah data lokal sinkron dengan remote
```
