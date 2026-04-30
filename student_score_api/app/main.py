from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from .schemas import StudentData
from .models import predictor

app = FastAPI(title="Student Score Prediction API")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
def predict_score(student: StudentData):
    pred = predictor.predict(student.dict())
    return {"prediction": pred}