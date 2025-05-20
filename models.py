# models.py
import os
import pickle
import numpy as np
import logging
import re
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)

def list_models_with_names(models_folder='models'):
    # Only show .pkl files, skip scaler if you wish
    model_files = [
        f for f in os.listdir(models_folder)
        if f.endswith('.pkl') and 'scaler' not in f.lower()
    ]

    def prettify(filename):
        name = os.path.splitext(filename)[0]
        name = re.sub(r'^top_model_\d+_', '', name)
        name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
        return name.strip()

    pretty_names = [prettify(f) for f in model_files]
    return model_files, pretty_names

def create_sample_models():
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('models/linear_regression.pkl'):
        X = np.random.rand(100, 5)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 5 * X[:, 3] + 0.5 * X[:, 4] + np.random.randn(100) * 0.5

        lr_model = LinearRegression()
        lr_model.fit(X, y)
        with open('models/linear_regression.pkl', 'wb') as f:
            pickle.dump(lr_model, f)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        with open('models/random_forest.pkl', 'wb') as f:
            pickle.dump(rf_model, f)

        scaler = StandardScaler()
        scaler.fit(X)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        logging.info("Sample models created and saved.")

def load_model(model_name):
    path = f'models/{model_name}.pkl'
    if not os.path.exists(path):
        logging.error(f"Model file '{model_name}' not found.")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict_man_months(features, model, scaler=None):
    if scaler:
        features_scaled = scaler.transform(features.reshape(1, -1))
    else:
        features_scaled = features.reshape(1, -1)

    prediction = model.predict(features_scaled)[0]
    logging.info(f"Prediction made with features: {features.tolist()} -> {prediction:.2f} man-months")
    return max(0, prediction)
