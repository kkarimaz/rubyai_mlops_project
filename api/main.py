from fastapi import FastAPI
from pydantic import BaseModel
import joblib

class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

app = FastAPI()
model = joblib.load("models/model.pkl")  # load model yang sudah disimpan dari proses training

LABEL_WINE = {
    0: "Barolo",
    1: "Grignolino",
    2: "Barbera"
}

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/health")
def read_health():
    return {"status": "healthy"}

# Outputs prediction based when given (hence the post) input data
@app.post("/predict")
def predict(features: WineFeatures):
    # Ekstrak nilai dari pelanggan dan jadikan List
    # features.dict().values() akan mengambil semua angka dari pesanan
    data_pesanan = list(features.model_dump().values())
    
    # Ubah ke format Array 2D 
    features_array = [data_pesanan] 
    
    # Prediksi
    prediction = model.predict(features_array)
    
    # Kembalikan Jawaban
    # PENTING: model.predict menghasilkan Numpy Array (contoh: [1]). 
    # Internet/JSON tidak paham Numpy Array, jadi kita harus ubah ke int biasa menggunakan .item()
    hasil_angka = int(prediction[0].item())
    hasil_teks = LABEL_WINE.get(hasil_angka, "Unknown Class")
    
    return {
        "message": "Prediction created!",
        "predicted_class_id": hasil_angka,
        "predicted_class_label": hasil_teks}