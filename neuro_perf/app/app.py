from model_utils import train_eval
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io

app = FastAPI(
    title="P90 Response Time Predictor API",
    description="Upload a CSV file containing Average Response Time, Latency, and 90th Percentile Response Time columns. The API trains a regression model and returns metrics and predictions.",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "Welcome to the P90 Response Time Prediction API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Train and evaluate model
        metrics, y_actual, y_pred = train_eval(df)

        return {
            "status": "success",
            "metrics": metrics,
            "predictions": {
                "actual": y_actual,
                "predicted": y_pred
            }
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
