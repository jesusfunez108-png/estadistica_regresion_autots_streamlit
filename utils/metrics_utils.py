"""
📈 Utilidades de Métricas
Calcula métricas estadísticas para evaluación de modelos
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Evitar división por cero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r_squared(y_true, y_pred):
    """Coeficiente de determinación R²"""
    return r2_score(y_true, y_pred)


def adjusted_r_squared(y_true, y_pred, n_features):
    """R² ajustado"""
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2


def calculate_regression_metrics(y_true, y_pred, n_features=None):
    """
    Calcula todas las métricas de regresión
    
    Returns:
        dict: Diccionario con todas las métricas
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'R²': r_squared(y_true, y_pred),
    }
    
    if n_features is not None:
        metrics['R² Ajustado'] = adjusted_r_squared(y_true, y_pred, n_features)
    
    return metrics


def calculate_classification_metrics(y_true, y_pred):
    """
    Calcula métricas para clasificación
    
    Returns:
        dict: Diccionario con métricas de clasificación
    """
    metrics = {
        'Exactitud': accuracy_score(y_true, y_pred),
        'Precisión': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    return metrics


def get_residuals(y_true, y_pred):
    """Calcula residuos"""
    return np.array(y_true) - np.array(y_pred)


def get_residual_statistics(residuals):
    """Estadísticas de residuos"""
    residuals = np.array(residuals)
    return {
        'Media': np.mean(residuals),
        'Desviación Estándar': np.std(residuals),
        'Mínimo': np.min(residuals),
        'Máximo': np.max(residuals),
        'Mediana': np.median(residuals),
    }
