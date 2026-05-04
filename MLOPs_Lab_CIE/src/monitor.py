import json
import numpy as np

temps = []
occ = []
preds = []

# Read logs
with open("logs/predictions.jsonl") as f:
    for line in f:
        d = json.loads(line)
        temps.append(d["input"]["temperature_c"])
        occ.append(d["input"]["occupancy_pct"])
        preds.append(d["prediction"])

temps = np.array(temps)
occ = np.array(occ)

# Given training means
train_temp_mean = 32.24
train_occ_mean = 57.52

live_temp_mean = np.mean(temps)
live_occ_mean = np.mean(occ)

# Calculate shifts
temp_shift = abs(live_temp_mean - train_temp_mean)
occ_shift = abs(live_occ_mean - train_occ_mean)

alerts = []

# Temperature drift
if temp_shift > 5.46 or live_temp_mean > train_temp_mean:
    alerts.append({
        "feature": "temperature_c",
        "train_mean": train_temp_mean,
        "live_mean": float(live_temp_mean),
        "shift": float(temp_shift),
        "threshold": 5.46,
        "status": "ALERT"
    })

# Occupancy drift
if occ_shift > 13.17 or live_occ_mean > train_occ_mean:
    alerts.append({
        "feature": "occupancy_pct",
        "train_mean": train_occ_mean,
        "live_mean": float(live_occ_mean),
        "shift": float(occ_shift),
        "threshold": 13.17,
        "status": "ALERT"
    })

# Final output
output = {
    "total_predictions": len(preds),
    "mean_prediction": float(np.mean(preds)),
    "drift_detected": len(alerts) > 0,
    "alerts": alerts
}

# Save result
with open("results/step3_s5.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 done")