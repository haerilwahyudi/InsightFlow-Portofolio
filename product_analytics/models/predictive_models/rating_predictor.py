 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

class AdvancedRatingPredictor:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.models = {
            'xgboost': xgb.XGBRegressor(),
            'lightgbm': lgb.LGBMRegressor(),
            'gboost': GradientBoostingRegressor()
        }
        
    def hyperparameter_tuning(self):
        """Grid search for optimal parameters"""
        param_grid = {
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            },
            # Add other model params...
        }
        
        self.best_models = {}
        for name, model in self.models.items():
            gs = GridSearchCV(
                model,
                param_grid[name],
                cv=5,
                scoring='neg_mean_squared_error'
            )
            gs.fit(self.X_train, self.y_train)
            self.best_models[name] = gs.best_estimator_
            
        return self.best_models
    
    def ensemble_prediction(self, X_test):
        """Combine predictions from multiple models"""
        predictions = []
        for name, model in self.best_models.items():
            pred = model.predict(X_test)
            predictions.append(pred)
            
        return np.mean(predictions, axis=0)
