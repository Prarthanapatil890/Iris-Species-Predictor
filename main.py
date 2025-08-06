from fastapi import FastAPI, HTTPException         # FastAPI for creating API endpoints and handling HTTP errors
from pydantic import BaseModel, Field              # BaseModel for request validation; Field to set input constraints
import joblib                                      # For loading the saved machine learning model
import numpy as np                                 # For numerical operations and input formatting

# Load the trained Random Forest model saved as a pickle file

model = joblib.load("iris_model.pkl")

# Class labels corresponding to prediction output

iris_classes = ["setosa", "versicolor", "virginica"]

# Create FastAPI app instance

app = FastAPI(title="Iris Species Predictor")

# Define the expected input format using Pydantic model
class IrisRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=10)   # Must be > 0 and < 10
    sepal_width: float = Field(..., gt=0, lt=10)
    petal_length: float = Field(..., gt=0, lt=10)
    petal_width: float = Field(..., gt=0, lt=10)

# Root endpoint to confirm the API is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API"}

# Prediction endpoint that accepts POST requests
@app.post("/predict")
def predict_species(data: IrisRequest):
    try:
        # Convert input values into a 2D array for the model
        input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
        
        # Make prediction using the loaded model
        prediction = model.predict(input_data)[0]

        # Convert numerical prediction to species name
        species = iris_classes[prediction]

        # Return prediction result
        return {"species": species}

    except Exception as e:
        # Handle and report any unexpected errors
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
