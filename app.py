from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title='Heart Disease Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

model = load(pathlib.Path('model/diabetesv1.joblib'))

class InputData(BaseModel):
    gender: int = 1      
    age: float = 54.0       
    hypertension: int = 0    
    heart_disease: int = 0   
    smoking_history: int = 0 
    bmi: float = 27.32
    HbA1c_level: float = 6.6
    blood_glucose_level: float = 140.0

class OutputData(BaseModel):
    score: float = 0.80318881046519

@app.post('/score', response_model=OutputData)
def score(data: InputData):
    model_input = np.array([v for k, v in data.dict().items()]).reshape(1, -1)
    #result = model.predict_proba(model_input)[:, -1]
    result = model.predict(model_input)
    print(result)
    return {'score': float(result[0])}