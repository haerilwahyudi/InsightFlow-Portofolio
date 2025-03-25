import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class RatingPredictorOptimizer:
    """Advanced rating prediction with hyperparameter optimization"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.study = None
        self.best_model = None
        
    def optimize_xgb(self, n_trials=100):
        """Bayesian optimization for XGBoost"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
            }
            
            model = xgb.XGBRegressor(**params)
            return cross_val_score(
                model, self.X, self.y,
                scoring='neg_mean_squared_error',
                cv=5
            ).mean()
            
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials)
        
        self.best_model = xgb.XGBRegressor(**self.study.best_params)
        self.best_model.fit(self.X, self.y)
        return self.best_model
        
    def create_ensemble(self):
        """Stacking ensemble of multiple models"""
        estimators = [
            ('xgb', xgb.XGBRegressor()),
            ('lgb', lgb.LGBMRegressor()),
            ('cat', cb.CatBoostRegressor(verbose=0))
        ]
        
        self.ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(),
            cv=5
        )
        self.ensemble.fit(self.X, self.y)
        return self.ensemble
        
    def explain_model(self, model=None):
        """SHAP explanations for model interpretability"""
        import shap
        if model is None:
            model = self.best_model
            
        explainer = shap.Explainer(model)
        shap_values = explainer(self.X)
        
        # Visualization
        shap.summary_plot(shap_values, self.X)
        return shap_values
