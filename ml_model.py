import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
dataset = pd.read_csv('train.csv')

# Optional: standardize column names
dataset.columns = dataset.columns.str.strip().str.lower()

# Feature selection – update this based on your dataset
# Example columns: temperature, humidity, windspeed, hour, weekday
X = dataset[['temperature', 'humidity', 'windspeed', 'hour', 'weekday']]
y = dataset['energy_consumption']  # Target column – change if needed

# Handle missing values
X = X.fillna(X.mean())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluation (optional)
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation:\nRMSE: {rmse:.2f}\nR2: {r2:.2f}")

# Save model and scaler
pickle.dump(model, open("ml_model.sav", "wb"))
pickle.dump(scaler, open("scaler.sav", "wb"))
