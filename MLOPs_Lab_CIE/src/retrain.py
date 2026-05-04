import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor

# Load datasets
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

# Combine data
combined_df = pd.concat([train_df, new_df], ignore_index=True)

# Extract X, y
X_train_orig = train_df.drop("energy_kwh", axis=1)
y_train_orig = train_df["energy_kwh"]

X_combined = combined_df.drop("energy_kwh", axis=1)
y_combined = combined_df["energy_kwh"]

# Split SAME test set (IMPORTANT)
X_train1, X_test, y_train1, y_test = train_test_split(
    X_train_orig, y_train_orig, test_size=0.2, random_state=42
)

X_train2, _, y_train2, _ = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

# Load best model name from Task 1
with open("results/step1_s1.json") as f:
    task1 = json.load(f)

best_model_name = task1["best_model"]

# Select model
if best_model_name == "Lasso":
    champion_model = Lasso()
    retrain_model = Lasso()
else:
    champion_model = GradientBoostingRegressor(random_state=42)
    retrain_model = GradientBoostingRegressor(random_state=42)

# Train champion (old data)
champion_model.fit(X_train1, y_train1)
champion_preds = champion_model.predict(X_test)
champion_rmse = np.sqrt(mean_squared_error(y_test, champion_preds))

# Train retrained model (combined data)
retrain_model.fit(X_train2, y_train2)
retrain_preds = retrain_model.predict(X_test)
retrained_rmse = np.sqrt(mean_squared_error(y_test, retrain_preds))

# Compare
improvement = champion_rmse - retrained_rmse
threshold = 0.5

if improvement >= threshold:
    action = "promoted"
    os.makedirs("models", exist_ok=True)
    import joblib
    joblib.dump(retrain_model, "models/retrained_model.pkl")
else:
    action = "kept_champion"

# Save output
output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_rmse": float(champion_rmse),
    "retrained_rmse": float(retrained_rmse),
    "improvement": float(improvement),
    "min_improvement_threshold": threshold,
    "action": action,
    "comparison_metric": "rmse"
}

os.makedirs("results", exist_ok=True)
with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed")