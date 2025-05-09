import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load and prepare data
dataset = pd.read_csv('train.csv')
dataset = dataset.rename(columns=lambda x: x.strip())

# Feature engineering
dataset['TimeOfDay'] = pd.cut(dataset['Hour'], 
                            bins=[-1, 6, 12, 18, 24],
                            labels=['Night', 'Morning', 'Afternoon', 'Evening'])
dataset['OccupancyDensity'] = dataset['Occupancy'] / dataset['SquareFootage']
dataset['ComfortIndex'] = dataset['Temperature'] * dataset['Humidity'] / 100
dataset['IsWeekend'] = dataset['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)

# Convert categorical variables
dataset['DayOfWeek'] = dataset['DayOfWeek'].astype('category')
dataset['Holiday'] = dataset['Holiday'].map({'Yes': 1, 'No': 0})
dataset['HVACUsage'] = dataset['HVACUsage'].map({'On': 1, 'Off': 0})
dataset['LightingUsage'] = dataset['LightingUsage'].map({'On': 1, 'Off': 0})

# Prepare features and target
X = dataset.drop(columns=['EnergyConsumption'])
y = dataset['EnergyConsumption']

# Define feature types
numeric_features = ['Month', 'Hour', 'Temperature', 'Humidity', 
                   'SquareFootage', 'Occupancy', 'RenewableEnergy',
                   'OccupancyDensity', 'ComfortIndex']
categorical_features = ['TimeOfDay', 'IsWeekend']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', PowerTransformer(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)])

# Create model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=50,
        random_state=42
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:\nRMSE: {rmse:.2f}\nR2: {r2:.2f}")

# Save the complete pipeline
pickle.dump(model, open("model/energy_model.pkl", "wb"))