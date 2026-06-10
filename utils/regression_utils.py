"""
📊 Utilidades de Regresión
Modelos de regresión lineal, polinómica y logística
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy import stats


class LinearRegressor:
    """Regresión Lineal Simple y Múltiple"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        """Entrena el modelo"""
        self.X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
        self.y = np.array(y)
        self.model.fit(self.X, self.y)
        
    def predict(self, X):
        """Realiza predicciones"""
        X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
        return self.model.predict(X)
    
    def get_params(self):
        """Obtiene parámetros del modelo"""
        if self.X.shape[1] == 1:
            return {
                'Intercepción (β₀)': self.model.intercept_,
                'Pendiente (β₁)': self.model.coef_[0]
            }
        else:
            params = {'Intercepción (β₀)': self.model.intercept_}
            for i, coef in enumerate(self.model.coef_):
                params[f'Coeficiente β{i+1}'] = coef
            return params
    
    def get_equation(self):
        """Retorna la ecuación del modelo"""
        if self.X.shape[1] == 1:
            return f"y = {self.model.intercept_:.4f} + {self.model.coef_[0]:.4f}*x"
        else:
            eq = f"y = {self.model.intercept_:.4f}"
            for i, coef in enumerate(self.model.coef_):
                eq += f" + {coef:.4f}*x{i+1}"
            return eq


class PolynomialRegressor:
    """Regresión Polinómica"""
    
    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()
        self.X_poly = None
        
    def fit(self, X, y):
        """Entrena el modelo"""
        X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
        self.X_poly = self.poly_features.fit_transform(X)
        self.model.fit(self.X_poly, y)
        
    def predict(self, X):
        """Realiza predicciones"""
        X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
        X_poly = self.poly_features.transform(X)
        return self.model.predict(X_poly)
    
    def get_degree(self):
        """Retorna el grado del polinomio"""
        return self.degree


class LogisticRegressor:
    """Regresión Logística para Clasificación"""
    
    def __init__(self, max_iter=1000):
        self.model = LogisticRegression(max_iter=max_iter)
        self.scaler = StandardScaler()
        self.X_scaled = None
        
    def fit(self, X, y):
        """Entrena el modelo"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X_scaled = self.scaler.fit_transform(X)
        self.model.fit(self.X_scaled, y)
        
    def predict(self, X):
        """Realiza predicciones"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Retorna probabilidades"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class RidgeRegressor:
    """Regresión Ridge (L2 Regularization)"""
    
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        
    def fit(self, X, y):
        """Entrena el modelo"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.model.fit(X, y)
        
    def predict(self, X):
        """Realiza predicciones"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model.predict(X)


class LassoRegressor:
    """Regresión Lasso (L1 Regularization)"""
    
    def __init__(self, alpha=0.1):
        self.model = Lasso(alpha=alpha)
        
    def fit(self, X, y):
        """Entrena el modelo"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.model.fit(X, y)
        
    def predict(self, X):
        """Realiza predicciones"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model.predict(X)


def pearson_correlation(x, y):
    """Calcula correlación de Pearson"""
    correlation, p_value = stats.pearsonr(x, y)
    return correlation, p_value


def spearman_correlation(x, y):
    """Calcula correlación de Spearman"""
    correlation, p_value = stats.spearmanr(x, y)
    return correlation, p_value


def test_linearity(residuals):
    """Test de linealidad (Ramsey RESET)"""
    # Aproximación simple: verificar si residuos son normales
    stat, p_value = stats.shapiro(residuals)
    return {
        'estadístico': stat,
        'p_valor': p_value,
        'lineal': p_value > 0.05
    }


def test_homoscedasticity(y_true, y_pred):
    """Test de homocedasticidad (Breusch-Pagan)"""
    residuals = np.array(y_true) - np.array(y_pred)
    resid_squared = residuals ** 2
    
    # Correlación entre residuos² y valores predichos
    correlation, p_value = stats.pearsonr(y_pred, resid_squared)
    
    return {
        'correlación': correlation,
        'p_valor': p_value,
        'homocedástico': p_value > 0.05
    }


def test_normality(residuals):
    """Test de normalidad de residuos"""
    stat, p_value = stats.shapiro(residuals)
    return {
        'estadístico': stat,
        'p_valor': p_value,
        'normal': p_value > 0.05
    }


def get_confidence_interval(y_true, y_pred, confidence=0.95):
    """Calcula intervalo de confianza"""
    residuals = np.array(y_true) - np.array(y_pred)
    std_error = np.std(residuals)
    
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * std_error
    
    lower = np.array(y_pred) - margin
    upper = np.array(y_pred) + margin
    
    return lower, upper
