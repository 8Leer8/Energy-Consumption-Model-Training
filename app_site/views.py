from django.shortcuts import render
import pickle
import pandas as pd
import numpy as np

def home(request):
    return render(request, 'prediction/index.html')

def getPredictions(month, hour, day_of_week, holiday, temperature, humidity, 
                  square_footage, occupancy, hvac_usage, lighting_usage, renewable_energy):
    # Load model and scaler
    model = pickle.load(open('model/ml_model.sav', 'rb'))
    scaler = pickle.load(open('model/scaler.sav', 'rb'))
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
    input_data['DayOfWeek'] = input_data['DayOfWeek'].astype('category').cat.codes
    input_data['Holiday'] = input_data['Holiday'].astype('category').cat.codes
    input_data['HVACUsage'] = input_data['HVACUsage'].astype('category').cat.codes
    input_data['LightingUsage'] = input_data['LightingUsage'].astype('category').cat.codes
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    return prediction[0]

def result(request):
    if request.method == 'POST':
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
        
        return render(request, 'result.html', {'result': result})
    
    return render(request, 'prediction/index.html')