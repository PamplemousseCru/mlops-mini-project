import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = joblib.load("regression.joblib")

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
text_model = T5ForConditionalGeneration.from_pretrained(model_name)

app = FastAPI()

class Numbers(BaseModel):
    size: int
    bedrooms: int
    garden: int


class TextInput(BaseModel):
    text: str


@app.post("/predict")
async def model_predict(numbers: Numbers):
    result = model.predict([[numbers.size, numbers.bedrooms, numbers.garden]])
    return {"y_pred": result[0]}


@app.post("/llm")
async def text_model_predict(input: TextInput):
    input_ids = tokenizer.encode(input.text, return_tensors="pt")
    outputs = text_model.generate(input_ids)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"prediction": decoded_output}
