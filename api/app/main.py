import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import datetime as dt

current_date = dt.datetime.now().strftime('%d-%m')

with open(f'models/model_{current_date}.pkl', 'rb') as f:
    model = pickle.load(f)
with open(f'models/vectorizer_{current_date}.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define input and output data types
class InputData(BaseModel):
    text: str

class OutputData(BaseModel):
    label: str

app = FastAPI()

@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    text = input_data.text
    X = vectorizer.transform([text])

    pred = model.predict(X)
    return {"label": pred[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
