# Energy Consumption Prediction System

## Project Overview

The Energy Consumption Prediction System is a web application that uses machine learning to predict building energy consumption based on various environmental and usage factors. The system provides accurate forecasts and insights to help users optimize their energy usage and reduce costs. Built with Django and modern web technologies, it offers an intuitive interface for both individual users and businesses to make data-driven energy management decisions.

## Technical Architecture

The project consists of two main components:

1. **Data Science Pipeline**: A machine learning pipeline that processes energy consumption data, trains multiple models, and selects the best performing one for deployment. This pipeline includes data cleaning, feature engineering, model training, and evaluation.

2. **Web Application**: A Django-based web application that provides a user-friendly interface for users to input building parameters, view predictions, and analyze energy consumption patterns.

### System Requirements

- Python 3.8+
- Django 4.2+
- scikit-learn
- pandas
- numpy
- plotly
- Tailwind CSS

## Data Science Pipeline

### Dataset

The project uses energy consumption data that includes the following features:

- Month (1-12)
- Hour of day (0-23)
- Temperature (°C)
- Humidity (%)
- Square footage (sq ft)
- Occupancy (number of people)
- Energy consumption (kWh) - target variable

### Data Preprocessing

The data preprocessing steps include:

- Handling missing values
- Removing outliers using IQR method
- Feature engineering:
  - Creating time-based features (season, day of week)
  - Normalizing numerical features using StandardScaler
  - Encoding categorical variables using One-Hot Encoding
  - Creating interaction features between temperature and occupancy

### Model Training

We trained and compared several machine learning models:

- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression

Each model was evaluated using:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² score
- Cross-validation scores

The best performing model was selected for deployment in the web application.

## Web Application

### Technology Stack

- **Backend**: 
  - Django (Python web framework)
  - Django REST framework for API endpoints
  - Django authentication system
- **Frontend**: 
  - HTML5, CSS3, JavaScript
  - Tailwind CSS for responsive design
  - Font Awesome for icons
- **Database**: 
  - SQLite (development)
  - PostgreSQL (production-ready)
- **Data Visualization**: 
  - Plotly.js for interactive charts
  - Custom dashboard components

### Key Features

- **User Authentication**:
  - Secure registration and login
  - Password reset functionality
  - User profile management
- **Prediction Form**:
  - Intuitive input fields for building parameters
  - Real-time validation
  - Mobile-responsive design
- **Results Page**:
  - Clear prediction display
  - Confidence intervals
  - Historical comparison
- **Dashboard**:
  - Real-time data visualization
  - Interactive charts
  - Export functionality
- **Responsive Design**:
  - Mobile-first approach
  - Cross-browser compatibility
  - Accessible interface

### Application Structure

The Django application follows the standard MVT (Model-View-Template) architecture:

- **Models**:
  - UserProfile: User information and preferences
  - Prediction: Prediction results and parameters
  - BuildingProfile: Building-specific information
- **Views**:
  - Class-based views for CRUD operations
  - Form handling and validation
  - API endpoints
- **Templates**:
  - Base template with common elements
  - Modular component design
  - Responsive layouts
- **Forms**:
  - Custom form validation
  - Dynamic field generation
  - Error handling

## Implementation Details

### Models

The application uses three main models:

- **UserProfile**: 
  - User authentication details
  - Preferences and settings
  - Usage history
- **Prediction**: 
  - Input parameters
  - Prediction results
  - Timestamp and metadata
- **BuildingProfile**:
  - Building characteristics
  - Historical data
  - Optimization settings

### Prediction Process

When a user submits the prediction form:

1. The form data is validated and processed
2. The trained machine learning model is loaded
3. Feature engineering is applied to the input data
4. The prediction is made with confidence intervals
5. Results are saved to the database
6. The user is redirected to the results page

### Dashboard Features

The dashboard provides:

- **Statistics**:
  - Total predictions made
  - Average consumption
  - Median consumption
  - Maximum consumption
- **Model Performance**:
  - RMSE, MAE, and R² metrics
  - Model comparison charts
  - Feature importance analysis
- **Consumption Patterns**:
  - Monthly trends
  - Hourly patterns
  - Seasonal analysis
- **Latest Predictions**:
  - Recent results
  - Input parameters
  - Confidence levels

## Deployment Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/energy-consumption-predictor.git
   cd energy-consumption-predictor
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. Run migrations:
   ```bash
   python manage.py migrate
   ```

6. Create superuser:
   ```bash
   python manage.py createsuperuser
   ```

7. Run the server:
   ```bash
   python manage.py runserver
   ```

## Future Enhancements

- **Model Improvements**:
  - Deep learning models
  - Time series analysis
  - Ensemble methods
- **User Features**:
  - Multiple building profiles
  - Custom reporting
  - API access
- **Analysis Tools**:
  - Energy optimization recommendations
  - Cost analysis
  - Environmental impact assessment
- **Integration**:
  - Weather API integration
  - Smart meter data
  - Building management systems
- **Visualization**:
  - 3D building models
  - Real-time monitoring
  - Custom chart types
- **Export Options**:
  - PDF reports
  - CSV data export
  - API endpoints

## Conclusion

The Energy Consumption Prediction System demonstrates the practical application of machine learning in energy management. By combining a robust data science pipeline with a user-friendly web interface, the system provides valuable insights to help users understand and optimize their energy consumption patterns.

The project showcases the integration of data science and web development, creating a complete end-to-end solution that delivers real value to users in terms of energy efficiency and cost savings.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 