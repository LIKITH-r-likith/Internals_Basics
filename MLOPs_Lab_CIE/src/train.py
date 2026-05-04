import pandas as pd
import numpy as np
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json

# Load dataset
data = pd.read_csv("data/training_data.csv")

X = data.drop("energy_kwh", axis=1)
y = data["energy_kwh"]

# Split (IMPORTANT RULE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow setup
mlflow.set_experiment("powergrid-energy-kwh")

models = {
    "Lasso": Lasso(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

results = []

# Train both models
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("experiment_type", "baseline_comparison")

        results.append({
            "name": name,
            "mae": float(mae),
            "rmse": float(rmse)
        })

# Select best model by MAE
best = min(results, key=lambda x: x["mae"])

# Train best model again
if best["name"] == "Lasso":
    final_model = Lasso()
else:
    final_model = GradientBoostingRegressor(random_state=42)

final_model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/best_model.pkl")

# Save JSON output
output = {
    "experiment_name": "powergrid-energy-kwh",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed")