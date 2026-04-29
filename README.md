---
title: Wine MLOps API
emoji: 🍷
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
---

# Wine Recognition MLOps Project

This project is an **end-to-end MLOps** implementation for wine variety classification using the UCI Wine dataset. It demonstrates a complete Machine Learning lifecycle — from data versioning, experiment tracking, to deployment using Docker and Hugging Face Spaces.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [How to Run](#how-to-run)
  - [1. Setup Environment](#1-setup-environment)
  - [2. Set Up Data with DVC](#2-set-up-data-with-dvc)
  - [3. Training & Experiment Tracking](#3-training--experiment-tracking)
  - [4. Run API Locally](#4-run-api-locally)
  - [5. Run with Docker](#5-run-with-docker)
- [API Documentation](#api-documentation)
  - [GET /](#get-)
  - [GET /health](#get-health)
  - [POST /predict](#post-predict)
- [Dataset & Model](#dataset--model)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [DVC Data Versioning](#dvc-data-versioning)

---

## Overview

This project classifies wine samples into 3 Italian cultivar types based on 13 chemical features:

| Label ID | Cultivar Name |
|----------|---------------|
| 0        | Barolo        |
| 1        | Grignolino    |
| 2        | Barbera       |

**Implemented MLOps Capabilities:**

- **Data Versioning** — DVC + DagsHub Storage ensures the dataset is reproducible in any environment.
- **Experiment Tracking** — MLflow automatically logs hyperparameters, metrics (Accuracy & F1-Score), and model artifacts for every run.
- **Model Registry** — The best model is saved as a `.pkl` file locally and logged to MLflow Artifacts.
- **Production API** — FastAPI with input validation using Pydantic V2 to serve predictions.
- **Containerization** — Docker image ready to deploy to any server or cloud platform such as Hugging Face Spaces.

---

## System Architecture

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

| Component           | Technology             | Version    |
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

## Folder Structure

```
rubyai_mlops_project/
│
├── api/
│   └── main.py             # FastAPI app: endpoint /, /health, /predict
│
├── data/
│   ├── wine.csv            # Local dataset (ignored by Git, tracked by DVC)
│   └── wine.csv.dvc        # DVC pointer file (tracked by Git)
│
├── models/
│   └── model.pkl           # Best model from training (ignored by Git)
│
├── src/
│   ├── prepare_data.py     # Fetches dataset from scikit-learn, saves to CSV
│   └── train.py            # Training pipeline + MLflow logging
│
├── .dvc/
│   └── config              # DVC remote configuration (DagsHub)
│
├── Dockerfile              # Build image for deployment (port 7860)
├── requirements.txt        # Development dependencies (includes DVC, MLflow)
├── requirements-prod.txt   # Production dependencies (without DVC, MLflow)
└── README.md
```

---

## How to Run

### 1. Setup Environment

**Clone the repository and install dependencies:**

```bash
git clone https://dagshub.com/kkarimaz/rubyai_mlops_project.git
cd rubyai_mlops_project

python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

---

### 2. Set Up Data with DVC

> **Option A — Pull from DagsHub (recommended):**
>
> Make sure you have access to the DagsHub remote. Run:
>
> ```bash
> dvc pull
> ```
>
> This will download `data/wine.csv` from DagsHub Storage.

> **Option B — Regenerate from source:**
>
> ```bash
> python src/prepare_data.py
> ```
>
> This script fetches the dataset directly from `sklearn.datasets.load_wine` and saves it to `data/wine.csv`.

---

### 3. Training & Experiment Tracking

```bash
python src/train.py
```

This script will:
1. Read `data/wine.csv`
2. Train **3 variations** of `RandomForestClassifier` with different hyperparameter combinations:

   | Variation | `n_estimators` | `max_depth` | Description       |
   |-----------|----------------|-------------|-------------------|
   | 1         | 10             | 3           | Lightweight model |
   | 2         | 50             | 5           | Medium model      |
   | 3         | 100            | 10          | Complex model     |

3. Log hyperparameters, Accuracy, and F1-Score to **MLflow** (hosted on DagsHub).
4. Save the best model (based on F1-Score) to `models/model.pkl`.

**View experiment results at:**
`https://dagshub.com/kkarimaz/rubyai_mlops_project`

---

### 4. Run API Locally

> Make sure `models/model.pkl` exists (run training first).

```bash
uvicorn api.main:app --reload --port 8000
```

The API will run at `http://localhost:8000`.

Open the interactive documentation (Swagger UI) at:
`http://localhost:8000/docs`

---

### 5. Run with Docker

**Build the image:**

```bash
docker build -t wine-mlops .
```

**Run the container:**

```bash
docker run -p 7860:7860 wine-mlops
```

The API will run at `http://localhost:7860`.

> This image uses `requirements-prod.txt` which only contains libraries needed for serving (FastAPI, Scikit-Learn, Joblib, etc.) — without MLflow and DVC — to keep the image size smaller.

---

## API Documentation

### GET /

Simple health check.

**Response:**
```json
{
  "message": "Hello, World!"
}
```

---

### GET /health

Check service status.

**Response:**
```json
{
  "status": "healthy"
}
```

---

### POST /predict

Accepts 13 wine chemical features and returns a predicted cultivar class.

**Request Body (JSON):**

| Field                          | Type    | Description                                      |
|--------------------------------|---------|--------------------------------------------------|
| `alcohol`                      | float   | Alcohol content                                  |
| `malic_acid`                   | float   | Malic acid content                               |
| `ash`                          | float   | Ash content                                      |
| `alcalinity_of_ash`            | float   | Alkalinity of ash                                |
| `magnesium`                    | float   | Magnesium content                                |
| `total_phenols`                | float   | Total phenols                                    |
| `flavanoids`                   | float   | Flavanoid content                                |
| `nonflavanoid_phenols`         | float   | Non-flavanoid phenol content                     |
| `proanthocyanins`              | float   | Proanthocyanin content                           |
| `color_intensity`              | float   | Color intensity                                  |
| `hue`                          | float   | Hue                                              |
| `od280_od315_of_diluted_wines` | float   | OD280/OD315 ratio of diluted wines               |
| `proline`                      | float   | Proline content                                  |

**Example Request:**
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

**Example Response:**
```json
{
  "message": "Prediction created!",
  "predicted_class_id": 0,
  "predicted_class_label": "Barolo"
}
```

**Example with `curl`:**
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
- **Number of samples:** 178 (80/20 train/test split)
- **Number of features:** 13 chemical features
- **Number of classes:** 3 wine cultivars
- **Algorithm:** `RandomForestClassifier` (Scikit-Learn)
- **Evaluation metrics:** Accuracy & F1-Score (weighted)

---

## MLflow Experiment Tracking

All runs are recorded in the MLflow Tracking Server hosted on DagsHub:

```
https://dagshub.com/kkarimaz/rubyai_mlops_project.mlflow
```

Experiment name: `Wine_Classification_RF`

Each run logs:
- **Parameters:** `n_estimators`, `max_depth`
- **Metrics:** `accuracy`, `f1_score`
- **Artifacts:** model file `.pkl`

---

## DVC Data Versioning

The dataset is managed using DVC with remote storage on DagsHub:

```
https://dagshub.com/kkarimaz/rubyai_mlops_project.dvc
```

The `data/wine.csv.dvc` file is committed to Git as a pointer to the exact data version. The actual data file (`wine.csv`) is not pushed to Git, but to DagsHub Storage.

**Commonly used DVC commands:**

```bash
dvc pull          # Download data from remote
dvc push          # Upload data to remote
dvc status        # Check if local data is in sync with remote
```
