# from fastapi import FastAPI
# from pydantic import BaseModel, Field
# import joblib

# app = FastAPI()

# model = joblib.load("models/best_model.pkl")

# class InputData(BaseModel):
#     temperature_c: float = Field(..., ge=15, le=45)
#     building_sqm: float = Field(..., ge=50, le=500)
#     occupancy_pct: float = Field(..., ge=20, le=100)
#     is_weekday: int = Field(..., ge=0, le=1)

# @app.get("/status")
# def status():
#     return {
#         "status": "healthy",
#         "model_loaded": True
#     }

# @app.post("/estimate")
# def estimate(data: InputData):
#     X = [[
#         data.temperature_c,
#         data.building_sqm,
#         data.occupancy_pct,
#         data.is_weekday
#     ]]
#     prediction = model.predict(X)[0]
#     return {"prediction": float(prediction)} //

# from datetime import datetime
# import json
# import os

# @app.post("/estimate")
# def estimate(data: InputData):
#     X = [[
#         data.temperature_c,
#         data.building_sqm,
#         data.occupancy_pct,
#         data.is_weekday
#     ]]

#     prediction = model.predict(X)[0]

#     # 🔥 LOGGING PART
#     log_entry = {
#         "timestamp": str(datetime.now()),
#         "input": {
#             "temperature_c": data.temperature_c,
#             "building_sqm": data.building_sqm,
#             "occupancy_pct": data.occupancy_pct,
#             "is_weekday": data.is_weekday
#         },
#         "prediction": float(prediction)
#     }

#     os.makedirs("logs", exist_ok=True)
#     with open("logs/predictions.jsonl", "a") as f:
#         f.write(json.dumps(log_entry) + "\n")

#     return {"prediction": float(prediction)}

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from datetime import datetime
import json
import os

# ✅ CREATE APP FIRST
app = FastAPI()

# ✅ LOAD MODEL
model = joblib.load("models/best_model.pkl")

# ✅ INPUT SCHEMA
class InputData(BaseModel):
    temperature_c: float = Field(..., ge=15, le=45)
    building_sqm: float = Field(..., ge=50, le=500)
    occupancy_pct: float = Field(..., ge=20, le=100)
    is_weekday: int = Field(..., ge=0, le=1)

# ✅ HEALTH CHECK
@app.get("/status")
def status():
    return {
        "status": "healthy",
        "model_loaded": True
    }

# ✅ PREDICTION + LOGGING
@app.post("/estimate")
def estimate(data: InputData):
    X = [[
        data.temperature_c,
        data.building_sqm,
        data.occupancy_pct,
        data.is_weekday
    ]]

    prediction = model.predict(X)[0]

    # LOGGING
    log_entry = {
        "timestamp": str(datetime.now()),
        "input": {
            "temperature_c": data.temperature_c,
            "building_sqm": data.building_sqm,
            "occupancy_pct": data.occupancy_pct,
            "is_weekday": data.is_weekday
        },
        "prediction": float(prediction)
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": float(prediction)}