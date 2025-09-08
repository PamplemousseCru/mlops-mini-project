import joblib
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("regression.joblib")

app = FastAPI()


class Numbers(BaseModel):
    size: int
    bedrooms: int
    garden: int


@app.post("/predict")
async def model_predict(numbers: Numbers):
    result = model.predict([[numbers.size, numbers.bedrooms, numbers.garden]])
    return {"y_pred": result[0]}
