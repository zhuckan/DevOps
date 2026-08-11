import pandas as pd
import mlflow
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

mlflow.set_tracking_uri("file:./mlruns")

experiment_name = "Creditcard_Model_Tuning"
experiment = mlflow.get_experiment_by_name(experiment_name)
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

best_metric_col = runs_df[[col for col in runs_df.columns if col.startswith('metrics.')]].max().idxmax()
best_run = runs_df.loc[runs_df[best_metric_col].idxmax()]
best_run_id = best_run['run_id']

print(f"Загружаю модель из лучшего запуска: {best_run_id}")
print(f"Лучшая метрика: {best_metric_col} = {best_run[best_metric_col]:.4f}")

model_path = f"mlruns/0/{best_run_id}/artifacts/best_model/model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

class InputData(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Hour: int

class Prediction(BaseModel):
    pred: int
    prob: float

@app.get('/status')
def status():
    return "I'm OK"

@app.get('/version')
def version():
    return {"model_type": type(model).__name__, "is_pipeline": True}

@app.post('/predict', response_model=Prediction)
def predict(form: InputData):
    df = pd.DataFrame.from_dict([form.dict()])
    y_proba = model.predict_proba(df)[0][1]
    y_pred = int(model.predict(df)[0])
    return {"pred": y_pred, "prob": float(y_proba)}