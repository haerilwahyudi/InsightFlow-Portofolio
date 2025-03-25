import pandas as pd
from sqlalchemy import create_engine
import logging

class ProductDataLoader:
    """Handles all data loading operations with multiple source support"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.supported_sources = ['csv', 'sql', 'api']
        
    def load_data(self, source_type='csv', **kwargs):
        """
        Main data loading method with source switching
        Args:
            source_type: Type of data source (csv/sql/api)
            kwargs: Source-specific parameters
        Returns:
            Cleaned pandas DataFrame
        """
        try:
            if source_type == 'csv':
                return self._load_csv(kwargs.get('filepath'))
            elif source_type == 'sql':
                return self._load_sql(
                    kwargs.get('connection_string'),
                    kwargs.get('query')
                )
            elif source_type == 'api':
                return self._load_api(
                    kwargs.get('endpoint'),
                    kwargs.get('params')
                )
            else:
                raise ValueError(f"Unsupported source type. Choose from {self.supported_sources}")
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise

    def _load_csv(self, filepath):
        """CSV-specific loader with automatic date parsing"""
        df = pd.read_csv(
            filepath,
            parse_dates=['Manufacturing Date', 'Expiration Date'],
            infer_datetime_format=True
        )
        self._validate_data(df)
        return df

    def _load_sql(self, connection_string, query):
        """Database loader with connection pooling"""
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        self._validate_data(df)
        return df

    def _validate_data(self, df):
        """Data quality checks"""
        required_columns = ['Product ID', 'Product Name', 'Price', 'Stock Quantity']
        if not all(col in df.columns for col in required_columns):
            missing = set(required_columns) - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        if df.isnull().sum().sum() > 0:
            self.logger.warning("Data contains missing values - imputation recommended") 
