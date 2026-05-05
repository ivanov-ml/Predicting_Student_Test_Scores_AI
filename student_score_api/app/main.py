from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import pandas as pd
from .schemas import StudentData
from .pipeline import load_pipeline

app = FastAPI(title="Student Score Prediction API")
templates = Jinja2Templates(directory="templates")

# Загружаем пайплайн (теперь один файл вместо двух)
pipeline = load_pipeline('models/student_pipeline.pkl')

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
def predict_score(student: StudentData):
    # Преобразуем словарь в DataFrame
    input_df = pd.DataFrame([student.dict()])
    # Предсказание (масштабирование + кодирование + модель)
    prediction = pipeline.predict(input_df)[0]
    # Ограничиваем от 0 до 100
    prediction = max(0, min(100, prediction))
    return {"prediction": round(prediction, 2)}