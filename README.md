# NYC-Taxi-Fare-Prediction
Predicting New York City taxi fares using Machine Learning models (Linear Regression, Random Forest, Gradient Boosting, XGBoost) with MLflow tracking and AWS deployment.

# NYC Taxi Fare Prediction 🚖

This project predicts New York City taxi fares using machine learning models trained on the official [Kaggle NYC Taxi Fare dataset](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction).

It includes preprocessing, feature engineering, model training, hyperparameter tuning, and deployment on AWS using FastAPI.

---

## Models Trained
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Tuned Gradient Boosting (via RandomizedSearchCV)
- XGBoost Regressor

---

## Evaluation Metric
- **Root Mean Squared Error (RMSE)**  
- Best model RMSE: ~4.32

---

## MLflow Tracking
All models and experiments are logged using MLflow.

---

## Deployment Plan
- FastAPI backend to serve predictions
- Dockerized app for easy deployment
- Hosted on AWS EC2
- Simple HTML/CSS frontend to send requests to FastAPI

---

## 💼 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, MLflow
- FastAPI, Docker
- AWS EC2
- Git + GitHub

---

## 📁 Project Structure (coming soon)
