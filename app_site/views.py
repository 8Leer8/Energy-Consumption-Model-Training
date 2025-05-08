import pickle
import numpy as np
from django.shortcuts import render

# Load model and scaler
model = pickle.load(open('model/ml_model.sav', 'rb'))
scaler = pickle.load(open('model/scaler.sav', 'rb'))

def index(request):
    return render(request, 'prediction/index.html')

def predict(request):
    if request.method == 'POST':
        # Get form data
        day_of_week = int(request.POST.get('day_of_week'))
        holiday = int(request.POST.get('holiday'))
        hvac_usage = int(request.POST.get('hvac_usage'))
        lighting_usage = int(request.POST.get('lighting_usage'))

        # Prepare input for model
        input_data = np.array([[day_of_week, holiday, hvac_usage, lighting_usage]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        return render(request, 'prediction/index.html', {'prediction': prediction})

    return render(request, 'prediction/index.html')
