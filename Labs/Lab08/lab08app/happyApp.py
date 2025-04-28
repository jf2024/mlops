from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel

app = FastAPI(
    title="Happiness Score Predictor",
    description="Predict Happiness Score based on socio-economic indicators.",
    version="0.1",
)

# Root endpoint
@app.get('/')
def main():
    return {'message': 'This is a model for predicting Happiness Score'}

# Input schema
class RequestBody(BaseModel):
    Crime_Rate: float
    Unemployment_Rate: float
    Work_Life_Balance: float
    Freedom: float

# Load the model at startup
@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    model_pipeline = joblib.load("Ridge_best_model.joblib")

# Prediction endpoint
@app.post('/predict')
def predict(data: RequestBody):
    # Create a DataFrame in the correct format
    input_features = {
        'Crime_Rate': data.Crime_Rate,
        'Unemployment_Rate': data.Unemployment_Rate,
        'Work_Life_Balance': data.Work_Life_Balance,
        'Freedom': data.Freedom
    }
    
    import pandas as pd
    input_df = pd.DataFrame([input_features])

    # Make sure columns are in the expected order
    expected_order = model_pipeline.feature_names_in_
    input_df = input_df[expected_order]

    # Predict
    prediction = model_pipeline.predict(input_df)

    return {'Predicted Happiness Score': round(prediction[0], 2)}
