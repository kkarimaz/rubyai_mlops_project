import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import joblib
import dagshub

load_dotenv()
dagshub_token = os.getenv("DAGSHUB_TOKEN")

# Setup MLflow Tracking
os.environ["MLFLOW_TRACKING_USERNAME"] = "kkarimaz"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set URL Tracking DagsHub 
mlflow.set_tracking_uri("https://dagshub.com/kkarimaz/rubyai_mlops_project.mlflow")

dagshub_token = os.getenv("DAGSHUB_TOKEN")
# dagshub.init(repo_owner='kkarimaz', repo_name='rubyai_mlops_project', mlflow=True)

# Beri nama eksperimen
mlflow.set_experiment("Wine_Classification_Cloud_V2")

# Load data
print("Memuat data...")
df = pd.read_csv("data/wine.csv")

# Split data menjadi fitur (X) dan target (y)
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3 kombinasi hyperparameter: 3 ukuran hutan (n_estimators) dan kedalaman pohon (max_depth)
params_list = [
    {"n_estimators": 10, "max_depth": 3},   # Model ringan
    {"n_estimators": 50, "max_depth": 5},   # Model medium
    {"n_estimators": 100, "max_depth": 10}  # Model kompleks
]

best_f1 = 0
best_model = None

# Mulai proses training dan tracking
print("Memulai proses training dan tracking...")

for i, params in enumerate(params_list):
    # Membuka sesi (run) di MLflow untuk setiap kombinasi
    with mlflow.start_run(run_name=f"RF_Model_Variasi_{i+1}"):
        
        # Inisiasi dan Train Model
        rf = RandomForestClassifier(**params, random_state=42)
        rf.fit(X_train, y_train)
        
        # Prediksi
        y_pred = rf.predict(X_test)
        
        # Hitung Metrik
        acc = accuracy_score(y_test, y_pred)
        # Pakai average="weighted" karena targetnya ada 3 kelas (0 = Barolo, 1 = Grignolino, 2 = Barbera)
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        print(f"Run {i+1} | Params: {params} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        # Pencatatan di MLflow
        mlflow.log_params(params)                            # Catat Hyperparameter
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1}) # Catat Hasil Evaluasi
        mlflow.sklearn.log_model(rf, "model")                # Simpan File Modelnya (.pkl) di cloud
        
        # Cek apakah ini model terbaik sejauh ini
        if f1 > best_f1:
            best_f1 = f1
            best_model = rf

# Simpan model terbaik secara lokal
joblib.dump(best_model, "models/model.pkl")
print("\nTraining selesai! Model terbaik dengan F1-Score {:.4f} berhasil disimpan di models/model.pkl".format(best_f1))