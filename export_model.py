import mlflow
import joblib
import os

mlflow.set_tracking_uri("file:./mlruns")

experiment_name = "Creditcard_Model_Tuning"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    raise SystemExit(f"Experiment {experiment_name} not found. Run task1.py first.")

runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
if runs_df.empty:
    raise SystemExit("No runs found in MLflow experiment.")

metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
if not metric_cols:
    raise SystemExit("No metrics found.")


best_metric_col = metric_cols[0]
best_run = runs_df.loc[runs_df[best_metric_col].idxmax()]
best_run_id = best_run['run_id']

model_path = f"mlruns/0/{best_run_id}/artifacts/best_model/model.pkl"
if not os.path.exists(model_path):
    raise SystemExit(f"Model file not found: {model_path}")

model = joblib.load(model_path)
joblib.dump(model, "model.pkl")
print(f"Saved model.pkl from run {best_run_id}")
