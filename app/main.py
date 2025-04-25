from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Create the FastAPI app
app = FastAPI()

# Load your trained model
model = joblib.load("../models/gbr_model_v2.pkl")  # change if using another model

# Define the input data structure
class FarePredictionRequest(BaseModel):
    passenger_count: int
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    hour: int
    distance_km: float
    is_peak_hour: int

# Define the prediction route
@app.post("/predict")
def predict_fare(data: FarePredictionRequest):
    input_features = np.array([
        data.passenger_count,
        data.pickup_latitude,
        data.pickup_longitude,
        data.dropoff_latitude,
        data.dropoff_longitude,
        data.distance_km,
        data.hour,
        data.is_peak_hour
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)
    return {"predicted_fare": round(float(prediction[0]), 2)}
