import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
df = pd.read_csv('data/train.csv')

# Enhanced feature engineering
print("Performing feature engineering...")
def create_features(df):
    # Time-based features
    df['Season'] = pd.cut(df['Month'], 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['Winter', 'Spring', 'Summer', 'Fall'])
    df['Quarter'] = pd.cut(df['Month'], 
                          bins=[0, 3, 6, 9, 12], 
                          labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df['TimeOfDay'] = pd.cut(df['Hour'],
                            bins=[-1, 6, 12, 18, 24],
                            labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    # Environmental features
    df['ComfortIndex'] = df['Temperature'] * df['Humidity'] / 100
    df['TemperatureSquared'] = df['Temperature'] ** 2
    df['HumiditySquared'] = df['Humidity'] ** 2
    
    # Building features
    df['OccupancyDensity'] = df['Occupancy'] / df['SquareFootage']
    df['SquareFootagePerPerson'] = df['SquareFootage'] / df['Occupancy'].replace(0, 1)
    
    # Time patterns
    df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
    df['IsHoliday'] = df['Holiday'].astype(int)
    
    # Interaction features
    df['TempHumidityInteraction'] = df['Temperature'] * df['Humidity']
    df['OccupancyTempInteraction'] = df['Occupancy'] * df['Temperature']
    df['OccupancyHumidityInteraction'] = df['Occupancy'] * df['Humidity']
    
    # Usage patterns
    df['HVACUsage'] = df['HVACUsage'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df['LightingUsage'] = df['LightingUsage'].map({'Low': 0, 'Medium': 1, 'High': 2})
    
    return df

# Apply feature engineering
df = create_features(df)

# Split features and target
X = df.drop('EnergyConsumption', axis=1)
y = df['EnergyConsumption']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric and categorical features
numeric_features = ['Month', 'Hour', 'Temperature', 'Humidity', 'SquareFootage', 
                   'Occupancy', 'RenewableEnergy', 'ComfortIndex', 'TemperatureSquared',
                   'HumiditySquared', 'OccupancyDensity', 'SquareFootagePerPerson',
                   'TempHumidityInteraction', 'OccupancyTempInteraction', 
                   'OccupancyHumidityInteraction', 'HVACUsage', 'LightingUsage']

categorical_features = ['Season', 'Quarter', 'TimeOfDay', 'DayOfWeek', 'IsWeekend', 'IsHoliday']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define models to try
models = {
    'xgb': xgb.XGBRegressor(random_state=42),
    'rf': RandomForestRegressor(random_state=42),
    'gb': GradientBoostingRegressor(random_state=42)
}

# Define parameter grids for each model
param_grids = {
    'xgb': {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__subsample': [0.8, 0.9, 1.0],
        'regressor__colsample_bytree': [0.8, 0.9, 1.0]
    },
    'rf': {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    },
    'gb': {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__subsample': [0.8, 0.9, 1.0]
    }
}

# Train and evaluate models
best_score = float('-inf')
best_model = None
best_model_name = None

print("Training and evaluating models...")
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grids[model_name],
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{model_name.upper()} Results:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    if r2 > best_score:
        best_score = r2
        best_model = grid_search.best_estimator_
        best_model_name = model_name

print(f"\nBest model: {best_model_name.upper()}")
print(f"Best R² Score: {best_score:.4f}")

# Save the best model
print("\nSaving the best model...")
joblib.dump(best_model, 'model/energy_model.pkl')

# Save feature names for later use
feature_names = numeric_features + [f"{col}_{val}" for col, vals in 
                                  zip(categorical_features, 
                                      best_model.named_steps['preprocessor']
                                         .named_transformers_['cat']
                                         .categories_) 
                                  for val in vals]

# Save feature names
with open('model/feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_names))

print("Model training complete!") 