"""
⏳ Utilidades de Series de Tiempo
Análisis y modelado de series temporales
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler


def decompose_series(series, period=12):
    """
    Descompone serie temporal en componentes
    
    Args:
        series: Serie temporal
        period: Período (ej. 12 para mensual)
    
    Returns:
        dict: Componentes (tendencia, estacional, residual)
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Asegurar que tenemos un índice temporal
    if isinstance(series, pd.Series):
        data = series
    else:
        data = pd.Series(series)
    
    try:
        decomposition = seasonal_decompose(data, model='additive', period=period)
        
        return {
            'original': data.values,
            'tendencia': decomposition.trend.values,
            'estacional': decomposition.seasonal.values,
            'residual': decomposition.resid.values
        }
    except Exception as e:
        return {'error': str(e)}


def moving_average(series, window=5):
    """
    Calcula media móvil
    
    Args:
        series: Serie temporal
        window: Tamaño de ventana
    
    Returns:
        array: Media móvil
    """
    return pd.Series(series).rolling(window=window).mean().values


def exponential_smoothing(series, alpha=0.3):
    """
    Suavización exponencial
    
    Args:
        series: Serie temporal
        alpha: Factor de suavización (0-1)
    
    Returns:
        array: Serie suavizada
    """
    result = [series[0]]
    for i in range(1, len(series)):
        result.append(alpha * series[i] + (1 - alpha) * result[i-1])
    return np.array(result)


def difference_series(series, lag=1):
    """
    Diferencia la serie (para estacionariedad)
    
    Args:
        series: Serie temporal
        lag: Rezago
    
    Returns:
        array: Serie diferenciada
    """
    return np.array(series)[lag:] - np.array(series)[:-lag]


def adf_test(series):
    """
    Test Augmented Dickey-Fuller para estacionariedad
    
    Args:
        series: Serie temporal
    
    Returns:
        dict: Resultados del test
    """
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(series)
    
    return {
        'estadístico': result[0],
        'p_valor': result[1],
        'lags': result[2],
        'n_obs': result[3],
        'es_estacionaria': result[1] < 0.05
    }


def kpss_test(series):
    """
    Test KPSS para estacionariedad
    
    Args:
        series: Serie temporal
    
    Returns:
        dict: Resultados del test
    """
    from statsmodels.tsa.stattools import kpss
    
    try:
        result = kpss(series)
        return {
            'estadístico': result[0],
            'p_valor': result[1],
            'lags': result[2],
            'es_estacionaria': result[1] > 0.05
        }
    except Exception as e:
        return {'error': str(e)}


def acf_pacf(series, nlags=40):
    """
    Calcula ACF y PACF
    
    Args:
        series: Serie temporal
        nlags: Número de rezagos
    
    Returns:
        dict: ACF y PACF
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf, pacf
    
    acf_values = acf(series, nlags=nlags)
    pacf_values = pacf(series, nlags=nlags)
    
    return {
        'acf': acf_values,
        'pacf': pacf_values
    }


def arima_parameters(series):
    """
    Sugerencia de parámetros ARIMA (p, d, q)
    
    Args:
        series: Serie temporal
    
    Returns:
        dict: Parámetros sugeridos
    """
    # Test de diferenciación necesaria
    adf = adf_test(series)
    d = 0 if adf['es_estacionaria'] else 1
    
    # Test de segunda diferencia si es necesario
    if not adf['es_estacionaria']:
        diff_series = difference_series(series)
        adf2 = adf_test(diff_series)
        if adf2['es_estacionaria']:
            d = 1
        else:
            d = 2
    
    # ACF/PACF para p y q
    acf_pacf_vals = acf_pacf(series, nlags=20)
    acf = acf_pacf_vals['acf']
    pacf = acf_pacf_vals['pacf']
    
    # Contar picos significativos
    p = len([x for x in pacf[1:5] if abs(x) > 0.05])
    q = len([x for x in acf[1:5] if abs(x) > 0.05])
    
    return {
        'p': max(p, 1),
        'd': d,
        'q': max(q, 1),
        'sugerencia': f"ARIMA({max(p, 1)}, {d}, {max(q, 1)})"
    }


def fit_arima(series, order=(1, 1, 1)):
    """
    Ajusta modelo ARIMA
    
    Args:
        series: Serie temporal
        order: Tuple (p, d, q)
    
    Returns:
        modelo: Modelo ARIMA ajustado
    """
    from statsmodels.tsa.arima.model import ARIMA
    
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        return None


def forecast_arima(model, steps=10):
    """
    Realiza pronóstico con ARIMA
    
    Args:
        model: Modelo ARIMA ajustado
        steps: Pasos a pronosticar
    
    Returns:
        array: Pronóstico
    """
    try:
        forecast = model.get_forecast(steps=steps)
        return {
            'pronóstico': forecast.predicted_mean.values,
            'conf_int': forecast.conf_int().values
        }
    except Exception as e:
        return {'error': str(e)}


def seasonal_naive(series, season_length=12, steps=10):
    """
    Pronóstico naive estacional
    
    Args:
        series: Serie temporal
        season_length: Longitud de la estación
        steps: Pasos a pronosticar
    
    Returns:
        array: Pronóstico naive
    """
    series = np.array(series)
    forecast = []
    
    for i in range(steps):
        idx = len(series) - season_length + (i % season_length)
        forecast.append(series[idx])
    
    return np.array(forecast)


def get_autocorrelation(series, lag=1):
    """
    Calcula autocorrelación
    
    Args:
        series: Serie temporal
        lag: Rezago
    
    Returns:
        float: Autocorrelación
    """
    series = np.array(series)
    c0 = np.sum((series - np.mean(series)) ** 2) / len(series)
    c_lag = np.sum((series[:-lag] - np.mean(series)) * (series[lag:] - np.mean(series))) / len(series)
    return c_lag / c0


def ljung_box_test(series, nlags=10):
    """
    Test Ljung-Box para autocorrelación
    
    Args:
        series: Serie temporal
        nlags: Número de rezagos
    
    Returns:
        dict: Resultados del test
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    result = acorr_ljungbox(series, nlags=nlags)
    
    return {
        'estadístico': result[0].values,
        'p_valor': result[1].values,
        'hay_autocorrelación': np.any(result[1].values < 0.05)
    }
