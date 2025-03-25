from data_processing.data_loader import ProductDataLoader
from analytics.time_series import ProductDemandForecaster
from models.predictive_models.rating_predictor import AdvancedRatingPredictor

def main():
    # Load and prepare data
    loader = ProductDataLoader('products.csv')
    df = loader.load_data()
    
    # Time-series analysis
    forecaster = ProductDemandForecaster(df)
    arima_model = forecaster.fit_arima("Premium Laptop")
    forecast = arima_model.predict(start=0, end=30)
    
    # Predictive modeling
    X_train, X_test, y_train, y_test = train_test_split(...)
    predictor = AdvancedRatingPredictor(X_train, y_train)
    predictor.hyperparameter_tuning()
    predictions = predictor.ensemble_prediction(X_test)
    
    # Launch dashboard
    dashboard = ProductAnalyticsDashboard(df)
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 
