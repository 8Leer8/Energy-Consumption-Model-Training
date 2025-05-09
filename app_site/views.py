from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import logout
from django.contrib.auth.forms import UserCreationForm
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import plotly.utils
from django.core.cache import cache

def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully! You can now log in.')
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})
@login_required
def logout_view(request):
    logout(request)
    return redirect('home')
@login_required
def predict(request):
    months = [
        (1, "January"), (2, "February"), (3, "March"), (4, "April"),
        (5, "May"), (6, "June"), (7, "July"), (8, "August"),
        (9, "September"), (10, "October"), (11, "November"), (12, "December")
    ]
    hours = [
        (0, "12 AM"), (1, "1 AM"), (2, "2 AM"), (3, "3 AM"), (4, "4 AM"), (5, "5 AM"),
        (6, "6 AM"), (7, "7 AM"), (8, "8 AM"), (9, "9 AM"), (10, "10 AM"), (11, "11 AM"),
        (12, "12 PM"), (13, "1 PM"), (14, "2 PM"), (15, "3 PM"), (16, "4 PM"), (17, "5 PM"),
        (18, "6 PM"), (19, "7 PM"), (20, "8 PM"), (21, "9 PM"), (22, "10 PM"), (23, "11 PM")
    ]
    if request.method == 'POST':
        try:
            # Get all form data
            month = int(request.POST.get('month'))
            hour = int(request.POST.get('hour'))
            day_of_week = request.POST.get('day_of_week')
            holiday = request.POST.get('holiday')
            temperature = float(request.POST.get('temperature'))
            humidity = float(request.POST.get('humidity'))
            square_footage = float(request.POST.get('square_footage'))
            occupancy = int(request.POST.get('occupancy'))
            hvac_usage = request.POST.get('hvac_usage')
            lighting_usage = request.POST.get('lighting_usage')
            renewable_energy = float(request.POST.get('renewable_energy'))
            result = getPredictions(
                month, hour, day_of_week, holiday, temperature, humidity,
                square_footage, occupancy, hvac_usage, lighting_usage, renewable_energy
            )
            # Store the latest prediction in cache
            cache.set('latest_prediction', {
                'prediction': result,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'input_data': {
                    'month': month,
                    'hour': hour,
                    'temperature': temperature,
                    'humidity': humidity,
                    'square_footage': square_footage,
                    'occupancy': occupancy
                }
            })
            return render(request, 'predict.html', {'result': result, 'months': months, 'hours': hours})
        except Exception as e:
            messages.error(request, f'Error making prediction: {str(e)}')
    return render(request, 'predict.html', {'months': months, 'hours': hours})

def getPredictions(month, hour, day_of_week, holiday, temperature, humidity, 
                  square_footage, occupancy, hvac_usage, lighting_usage, renewable_energy):
    # Load model (preprocessing is included in the pipeline)
    model = pickle.load(open('model/energy_model.pkl', 'rb'))
    
    input_data = pd.DataFrame({
        'Month': [month],
        'Hour': [hour],
        'DayOfWeek': [day_of_week],
        'Holiday': [holiday],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'SquareFootage': [square_footage],
        'Occupancy': [occupancy],
        'HVACUsage': [hvac_usage],
        'LightingUsage': [lighting_usage],
        'RenewableEnergy': [renewable_energy]
    })

    # Feature engineering for prediction
    input_data['TimeOfDay'] = pd.cut(
        input_data['Hour'],
        bins=[-1, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening']
    )
    input_data['OccupancyDensity'] = input_data['Occupancy'] / input_data['SquareFootage']
    input_data['ComfortIndex'] = input_data['Temperature'] * input_data['Humidity'] / 100
    input_data['IsWeekend'] = input_data['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)

    # Make prediction (preprocessing is handled by the pipeline)
    prediction = model.predict(input_data)
    
    return prediction[0]

@login_required
def dashboard(request):
    try:
        # Load the dataset for visualization
        df = pd.read_csv('data/train.csv')

        # Calculate statistics
        total_records = len(df)
        avg_consumption = df['EnergyConsumption'].mean() if 'EnergyConsumption' in df else None
        max_consumption = df['EnergyConsumption'].max() if 'EnergyConsumption' in df else None
        min_consumption = df['EnergyConsumption'].min() if 'EnergyConsumption' in df else None
        median_consumption = df['EnergyConsumption'].median() if 'EnergyConsumption' in df else None

        # Load model and calculate performance metrics
        model = pickle.load(open('model/energy_model.pkl', 'rb'))
        xgb_model = model.named_steps['regressor']
        
        # Load test data and calculate performance metrics
        test_df = pd.read_csv('data/test.csv')
        
        # Apply feature engineering to test data
        test_df['TimeOfDay'] = pd.cut(
            test_df['Hour'],
            bins=[-1, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening']
        )
        test_df['OccupancyDensity'] = test_df['Occupancy'] / test_df['SquareFootage']
        test_df['ComfortIndex'] = test_df['Temperature'] * test_df['Humidity'] / 100
        test_df['IsWeekend'] = test_df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
        
        X_test = test_df.drop('EnergyConsumption', axis=1)
        y_test = test_df['EnergyConsumption']
        
        # Make predictions on test data
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create performance metrics visualization
        metrics_fig = go.Figure()
        metrics_fig.add_trace(go.Bar(
            x=['RMSE', 'MAE', 'R² Score'],
            y=[rmse, mae, r2],
            text=[f'{rmse:.2f}', f'{mae:.2f}', f'{r2:.2f}'],
            textposition='auto',
        ))
        metrics_fig.update_layout(
            title='Model Performance Metrics',
            yaxis_title='Score',
            height=400
        )
        metrics_data = json.dumps(metrics_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Monthly consumption chart
        if 'Month' in df and 'EnergyConsumption' in df:
            monthly_data_df = df.groupby('Month')['EnergyConsumption'].mean().reset_index()
            x_month = monthly_data_df['Month'].astype(int).tolist()
            y_month = monthly_data_df['EnergyConsumption'].astype(float).tolist()
            monthly_fig = go.Figure()
            monthly_fig.add_trace(go.Scatter(x=x_month, y=y_month, mode='lines+markers', name='Energy Consumption'))
            monthly_fig.update_layout(
                title='Average Energy Consumption by Month',
                xaxis_title='Month',
                yaxis_title='EnergyConsumption'
            )
            monthly_data = json.dumps(monthly_fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            monthly_data = json.dumps({"data": [], "layout": {"title": "No Data"}})

        # Hourly consumption chart
        if 'Hour' in df and 'EnergyConsumption' in df:
            hourly_data_df = df.groupby('Hour')['EnergyConsumption'].mean().reset_index()
            x_hour = hourly_data_df['Hour'].astype(int).tolist()
            y_hour = hourly_data_df['EnergyConsumption'].astype(float).tolist()
            hourly_fig = go.Figure()
            hourly_fig.add_trace(go.Scatter(x=x_hour, y=y_hour, mode='lines+markers', name='Energy Consumption'))
            hourly_fig.update_layout(
                title='Average Energy Consumption by Hour',
                xaxis_title='Hour',
                yaxis_title='EnergyConsumption'
            )
            hourly_data = json.dumps(hourly_fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            hourly_data = json.dumps({"data": [], "layout": {"title": "No Data"}})

        # Feature importance chart
        model = pickle.load(open('model/energy_model.pkl', 'rb'))
        xgb_model = model.named_steps['regressor']
        
        # Get feature names from the pipeline
        numeric_features = ['Month', 'Hour', 'Temperature', 'Humidity', 
                          'SquareFootage', 'Occupancy', 'RenewableEnergy',
                          'OccupancyDensity', 'ComfortIndex']
        categorical_features = ['TimeOfDay', 'IsWeekend']
        
        feature_names = numeric_features + list(model.named_steps['preprocessor']
                                     .named_transformers_['cat']
                                     .get_feature_names_out(categorical_features))
        
        importance = xgb_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True)
        
        feature_fig = go.Figure()
        feature_fig.add_trace(go.Bar(
            y=feature_importance_df['Feature'],
            x=feature_importance_df['Importance'],
            orientation='h'
        ))
        feature_fig.update_layout(
            title='Feature Importance in Energy Consumption Prediction',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=400
        )
        feature_data = json.dumps(feature_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Get latest prediction
        latest_prediction = cache.get('latest_prediction')

        context = {
            'total_records': total_records,
            'avg_consumption': round(avg_consumption, 2) if avg_consumption is not None else 'N/A',
            'max_consumption': round(max_consumption, 2) if max_consumption is not None else 'N/A',
            'min_consumption': round(min_consumption, 2) if min_consumption is not None else 'N/A',
            'median_consumption': round(median_consumption, 2) if median_consumption is not None else 'N/A',
            'monthly_data': monthly_data,
            'hourly_data': hourly_data,
            'feature_data': feature_data,
            'metrics_data': metrics_data,
            'model_metrics': {
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2': round(r2, 2)
            },
            'latest_prediction': latest_prediction,
            'monthly_x': x_month,
            'monthly_y': y_month,
            'hourly_x': x_hour,
            'hourly_y': y_hour,
        }

        return render(request, 'dashboard.html', context)
    except Exception as e:
        messages.error(request, f'Error loading dashboard: {str(e)}')
        return redirect('home')