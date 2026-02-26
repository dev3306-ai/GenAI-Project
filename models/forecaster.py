from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class SolarForecaster:
    def __init__(self, model_type='random_forest'):
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            self.model = LinearRegression()
        
        self.model_type = model_type
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'predictions': predictions
        }
    
    def get_feature_importance(self, feature_names):
        """Get feature importance for tree-based models"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(feature_names, self.model.feature_importances_))
        return None
