import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_eval(df: pd.DataFrame):
    required_cols = ["Average Response Time (ms)", "90th Percentile Response Time (ms)", "Throughput (requests/sec)", "Latency (ms)"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing required columns: {col}")

    #split features and target
    X = df.drop(["90th Percentile Response Time (ms)", "Timestamp"], axis=1)
    y = df["90th Percentile Response Time (ms)"]

    #split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "Mean Absolute Error (ms)": round(mae, 2),
        "Mean Squared Error (ms^2)": round(mse, 2),
        "Root Mean Squared Error (ms)": round(rmse, 2),
        "R² Score": round(r2, 3)
    }

    # Return metrics and predictions
    return metrics, list(y_test.values), list(y_pred)
