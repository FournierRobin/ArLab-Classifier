import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import datetime as dt
from clearml import Task

def get_model_and_vectorizer():
    current_date = dt.datetime.now().strftime('%d-%m')
    task_name = f"3 : Train model {current_date}"
    model_task = Task.get_task(project_name="ArLab Classifier", task_name=task_name)
    # on prend celui du jour d'avant si il n'existe pas
    while model_task is None and current_date != '01-01':
        previous_date = (dt.datetime.strptime(current_date, '%d-%m') - dt.timedelta(days=1)).strftime('%d-%m')
        task_name = f"3 : Train model {previous_date}"
        model_task = Task.get_task(project_name="ArLab Classifier", task_name=task_name)
        current_date = previous_date

    if model_task is not None:
        try:
            model = model_task.artifacts[f'ArLab Model {current_date}'].get()
            vectorizer = model_task.artifacts[f'vectorizer {current_date}'].get()
            return model, vectorizer
        except KeyError:
            pass
    raise ValueError(f"No model and vectorizer found for task {task_name}")

model, vectorizer = get_model_and_vectorizer()

class InputData(BaseModel):
    text: str

class OutputData(BaseModel):
    label: str

app = FastAPI()

@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    try:
        text = input_data.text
        X = vectorizer.transform([text])

        pred = model.predict(X)
        return {"label": pred[0]}
    except:
        return {"label": "Aucun model disponible"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
