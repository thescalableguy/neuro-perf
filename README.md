# P90 Response Time Predictor

A lightweight FastAPI-based Machine Learning microservice that predicts the 90th Percentile Response Time (P90) using 
Average Response Time, Latency and Throughput as input features.

# Features

*1. Upload CSV Dataset File with Average Response Time, Latency, Throughput, and P90 Response Time* <br>
*2. Automatically trains a Random Forest Regression Model* <br>
*3. Returns:* <br>
    *- Mean Absolute Error* <br>
    *- Mean Squared Error* <br>
    *- R2 Score* <br>
*3. No reconfig needed - works on any dataset matching column schema*
*4. Restful API built on FastAPI - clean, async and Swagger-enabled*

# Setup

1. Clone the repository

```
git clone https://github.com/thescalableguy/neuro-perf
```
2. Install Dependencies

```
pip install -r requirements.txt
```

3. Start the FastAPI Server

```
uvicorn app:app --port 9000 --reload
```

# Test the API

4. Use Swagger UI

```
Open your browser and go to http://127.0.0.1:9000/docs
```
