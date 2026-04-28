from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/health")
def read_health():
    return {"status": "healthy"}

# Outputs prediction based when given (hence the post) input data
@app.post("/predict")
def create_prediction():
    return {"message": "Prediction created!"}