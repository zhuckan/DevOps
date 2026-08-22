import mlflow
import mlflow.sklearn
import joblib

mlflow.set_tracking_uri("file:./mlruns")

experiment_name = "Creditcard_Model_Tuning"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    raise SystemExit(f"Experiment {experiment_name} not found. Run task1.py first.")

runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_auprc DESC"]
)
if runs_df.empty:
    raise SystemExit("No runs found in MLflow experiment.")

best_run_id = runs_df.iloc[0]["run_id"]
best_auprc = runs_df.iloc[0]["metrics.test_auprc"]
print(f"Best run: {best_run_id}, test_auprc={best_auprc:.4f}")

model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/best_model")
joblib.dump(model, "model.pkl")
print("Saved model.pkl")
