# PathSnap – API Documentation

## 1. Introduction

PathSnap uses FastAPI to provide backend functionality for trip evaluation and route reconstruction.

The API implementation is located in:

`src/api/main.py`

It allows the Streamlit user interface to communicate with the machine learning inference server.

## 2. Technologies

- FastAPI: Backend API framework.
- Pydantic: Request and response validation.
- Joblib: Loading trained machine learning models.
- Pandas: Processing trip data.
- NumPy: Numerical computation.
- HMMRouteDecoder: Sequential route reconstruction.

## 3. Starting the API Server

Run the following command from the project root:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000