import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
dataset = pd.read_csv('train.csv')
dataset = dataset.rename(columns=lambda x: x.strip().lower())

# Feature selection
dataset = dataset[['day_of_week', 'holiday', 'hvac_usage', 'lighting_usage', 'energy_consumption']]
dataset['day_of_week'] = pd.to_numeric(dataset['day_of_week'], errors='coerce')
dataset['holiday'] = pd.to_numeric(dataset['holiday'], errors='coerce')
dataset['hvac_usage'] = pd.to_numeric(dataset['hvac_usage'], errors='coerce')
dataset['lighting_usage'] = pd.to_numeric(dataset['lighting_usage'], errors='coerce')
dataset = dataset.fillna(dataset.mean())

X = dataset.drop(['energy_consumption'], axis=1)
y = dataset['energy_consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
sc = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation:\nRMSE: {rmse:.2f}\nR2: {r2:.2f}")

# Save model and scaler
pickle.dump(model, open("ml_model.sav", "wb"))
pickle.dump(sc, open("scaler.sav", "wb"))