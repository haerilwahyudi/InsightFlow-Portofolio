import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

class ProductDemandForecaster:
    def __init__(self, df):
        self.df = df
        self.preprocess_data()
        
    def preprocess_data(self):
        """Convert data into time-series format"""
        self.df['Date'] = pd.to_datetime(self.df['Manufacturing Date'])
        self.ts_data = self.df.groupby(['Date', 'Product Name'])['Stock Quantity'].sum().unstack()
        
    def fit_arima(self, product_name, order=(1,1,1)):
        """ARIMA implementation for single product"""
        model = ARIMA(self.ts_data[product_name], order=order)
        self.arima_model = model.fit()
        return self.arima_model
        
    def fit_prophet(self, product_name):
        """Facebook Prophet model"""
        prophet_df = self.ts_data[product_name].reset_index()
        prophet_df.columns = ['ds', 'y']
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(prophet_df)
        return model
        
    def evaluate_model(self, actual, predicted):
        return {
            'mae': mean_absolute_error(actual, predicted),
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
        } 
