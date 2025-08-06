# 🚀 FastAPI ML Deployment – Iris Classifier

This project demonstrates how to deploy a machine learning model using **FastAPI**, a modern, high-performance Python framework for building APIs.

## 📌 Project Overview

* **Model**: Random Forest Classifier trained on the Iris dataset
* **API Framework**: FastAPI with input validation using Pydantic
* **Endpoints**:

  * `GET /`: Welcome message
  * `POST /predict`: Predicts Iris species from flower measurements
* **Docs**: Auto-generated Swagger UI (`/docs`) and ReDoc (`/redoc`)

## 🧠 Key Learnings

* Structuring ML model APIs with FastAPI
* Validating input with Pydantic
* Serving predictions with Joblib
* Testing APIs using Swagger UI and Postman
* Optional: Docker for deployment consistency

## ▶️ How to Run

```
# Install dependencies
pip install fastapi uvicorn scikit-learn joblib numpy

# Run the app
uvicorn main:app --reload
```

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to test the API.

