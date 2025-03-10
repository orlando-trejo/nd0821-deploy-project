from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import numpy as np
from typing import Dict

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoder
with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Create app
app = FastAPI()

class CensusInput(BaseModel):
    """
    Pydantic model for an individual census record.
    Using 'alias' for fields with hyphens so they match the CSV columns exactly.
    Includes examples for FastAPI docs.
    """
    age: int = Field(..., example=39, alias="age")
    workclass: str = Field(..., example="State-gov", alias="workclass")
    fnlgt: int = Field(..., example=77516, alias="fnlgt")
    education: str = Field(..., example="Bachelors", alias="education")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical", alias="occupation")
    relationship: str = Field(..., example="Not-in-family", alias="relationship")
    race: str = Field(..., example="White", alias="race")
    sex: str = Field(..., example="Male", alias="sex")
    capital_gain: float = Field(..., example=2174, alias="capital-gain")
    capital_loss: float = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


@app.get("/")
def root() -> Dict[str, str]:
    """
    Root endpoint that returns a friendly greeting.
    """
    return {"message": "Welcome to the inference API!"}


@app.post("/predict")
def predict(data: CensusInput) -> Dict[str, int]:
    """
    POST endpoint that performs a prediction using our trained model.
    """
    # Numeric columns in the array
    numeric_values = np.array([[
        data.age,
        data.fnlgt,
        data.education_num,
        data.capital_gain,
        data.capital_loss,
        data.hours_per_week
    ]])

    # Categorical columns for encoding
    cat_values = [
        data.workclass,
        data.education,
        data.marital_status,
        data.occupation,
        data.relationship,
        data.race,
        data.sex,
        data.native_country
    ]
    cat_values_encoded = encoder.transform([cat_values])

    # Concatenate numeric and categorical arrays
    input_data_encoded = np.concatenate([numeric_values, cat_values_encoded], axis=1)

    # Model inference
    prediction = model.predict(input_data_encoded)
    return {"prediction": int(prediction[0])}