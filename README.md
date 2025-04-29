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

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, MLflow
- FastAPI
- AWS EC2
- Git + GitHub

---

## Project Structure
├── app/ # FastAPI backend and frontend.html
│ ├── main.py
│ └── frontend.html
├── models/ # Trained model(s)
│ └── gbr_model_v2.pkl
├── data/ # Data storage (empty or sample)
├── notebooks/ # Jupyter Notebooks for exploration
├── requirements.txt # Python dependencies
├── README.md

## Setup & Run Locally
1. Clone the repo:
   git clone https://github.com/your-username/NYC-Taxi-Fare-Prediction.git
   cd NYC-Taxi-Fare-Prediction
2. Create and activate virtual environment:
   python3 -m venv venv
   source venv/bin/activate
3. Install required packages:
   pip install -r requirements.txt
4. Run the FastAPI app:
   uvicorn app.main:app --reload
5. Open your browser and visit:
   http://127.0.0.1:8000/docs

## Deployment Info
The project is deployed on an AWS EC2 server using FastAPI.

- Backend is hosted on port 8000.
- Frontend is served via nginx on port 80.

Security group rules allow public access to ports 80 and 8000.

## Live Demo
Access the deployed project here:  
[Taxi Fare Prediction App](http://13.40.123.85)

Use `/docs` for API testing:  
[API Documentation](http://13.40.123.85:8000/docs)
