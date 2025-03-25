import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ProductFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering pipeline with scikit-learn compatibility"""
    
    def __init__(self, extract_dimensions=True, add_time_features=True):
        self.extract_dimensions = extract_dimensions
        self.add_time_features = add_time_features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if self.extract_dimensions:
            X = self._extract_dimensions(X)
            
        if self.add_time_features:
            X = self._add_time_features(X)
            
        X = self._add_business_features(X)
        
        return X
    
    def _extract_dimensions(self, X):
        """Parse product dimensions into numerical features"""
        dims = X['Product Dimensions'].str.split('x', expand=True)
        X[['Length_cm', 'Width_cm', 'Height_cm']] = dims.astype(float)
        X['Volume_cm3'] = X['Length_cm'] * X['Width_cm'] * X['Height_cm']
        X['Size_Category'] = pd.cut(
            X['Volume_cm3'],
            bins=[0, 1000, 5000, np.inf],
            labels=['Small', 'Medium', 'Large']
        )
        return X
    
    def _add_time_features(self, X):
        """Create temporal features from dates"""
        X['Shelf_Life_Days'] = (X['Expiration Date'] - X['Manufacturing Date']).dt.days
        X['Manufacture_Year'] = X['Manufacturing Date'].dt.year
        X['Manufacture_Month'] = X['Manufacturing Date'].dt.month
        X['Days_Since_Manufacture'] = (pd.Timestamp.now() - X['Manufacturing Date']).dt.days
        return X
    
    def _add_business_features(self, X):
        """Business-specific derived features"""
        X['Inventory_Value'] = X['Price'] * X['Stock Quantity']
        X['Price_Per_Volume'] = X['Price'] / X['Volume_cm3']
        X['Stock_Risk'] = np.where(
            X['Days_Since_Manufacture'] > 0.8 * X['Shelf_Life_Days'],
            'High',
            np.where(
                X['Days_Since_Manufacture'] > 0.5 * X['Shelf_Life_Days'],
                'Medium',
                'Low'
            )
        )
        return X 
