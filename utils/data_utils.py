"""
📦 Utilidades de Datos
Carga, limpieza y procesamiento de datos
"""

import pandas as pd
import numpy as np
import io


def load_csv(file):
    """Carga archivo CSV"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise Exception(f"Error al cargar CSV: {e}")


def load_excel(file):
    """Carga archivo Excel"""
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        raise Exception(f"Error al cargar Excel: {e}")


def load_file(file):
    """Carga archivo CSV o Excel"""
    if file.name.endswith('.csv'):
        return load_csv(file)
    elif file.name.endswith(('.xlsx', '.xls')):
        return load_excel(file)
    else:
        raise ValueError("Formato no soportado. Use CSV o Excel.")


def create_sample_data():
    """
    Crea datos de ejemplo para demostración
    
    Returns:
        dict: Diccionario con diferentes datasets de ejemplo
    """
    np.random.seed(42)
    
    # 1. Datos de caudales de ríos
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    caudal = 100 + 30 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 5, 365)
    
    df_caudal = pd.DataFrame({
        'Fecha': dates,
        'Caudal (m³/s)': np.maximum(caudal, 10),
        'Precipitación (mm)': np.random.gamma(2, 2, 365),
        'Temperatura (°C)': 15 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 2, 365)
    })
    
    # 2. Datos de producción agrícola
    df_agricola = pd.DataFrame({
        'Año': np.arange(2010, 2024),
        'Producción (toneladas)': [100 + i*5 + np.random.normal(0, 5) for i in range(14)],
        'Área (hectáreas)': [50 + i*2 + np.random.normal(0, 2) for i in range(14)],
        'Lluvia (mm)': np.random.uniform(400, 800, 14),
        'Temperatura (°C)': np.random.uniform(18, 28, 14)
    })
    
    # 3. Datos de temperatura
    df_temperatura = pd.DataFrame({
        'Fecha': dates,
        'Temperatura Máxima (°C)': 25 + 8 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 1, 365),
        'Temperatura Mínima (°C)': 15 + 5 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 1, 365),
        'Humedad (%)': 60 + 20 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 5, 365)
    })
    
    # 4. Datos simulados para regresión
    X_sim = np.random.uniform(0, 100, 100)
    y_sim = 2 + 0.5 * X_sim + np.random.normal(0, 10, 100)
    
    df_regresion = pd.DataFrame({
        'X': X_sim,
        'Y': y_sim
    })
    
    return {
        'Caudales de Ríos': df_caudal,
        'Producción Agrícola': df_agricola,
        'Temperatura': df_temperatura,
        'Regresión Simulada': df_regresion
    }


def get_sample_data(nombre):
    """Obtiene un dataset de ejemplo específico"""
    datasets = create_sample_data()
    return datasets.get(nombre, None)


def clean_data(df):
    """
    Limpieza básica de datos
    
    Args:
        df: DataFrame
    
    Returns:
        tuple: (DataFrame limpio, dict con estadísticas de limpieza)
    """
    stats = {
        'filas_iniciales': len(df),
        'nulos_por_columna': df.isnull().sum().to_dict(),
        'duplicados': df.duplicated().sum()
    }
    
    # Eliminar duplicados
    df = df.drop_duplicates()
    
    # Eliminar filas con valores nulos
    df = df.dropna()
    
    stats['filas_finales'] = len(df)
    stats['filas_eliminadas'] = stats['filas_iniciales'] - stats['filas_finales']
    
    return df, stats


def get_numeric_columns(df):
    """Obtiene columnas numéricas"""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df):
    """Obtiene columnas categóricas"""
    return df.select_dtypes(include=['object']).columns.tolist()


def normalize_data(data):
    """
    Normaliza datos entre 0 y 1
    
    Args:
        data: Array o Series
    
    Returns:
        array: Datos normalizados
    """
    data = np.array(data)
    return (data - data.min()) / (data.max() - data.min())


def standardize_data(data):
    """
    Estandariza datos (media=0, std=1)
    
    Args:
        data: Array o Series
    
    Returns:
        array: Datos estandarizados
    """
    data = np.array(data)
    return (data - data.mean()) / data.std()


def train_test_split_data(df, test_size=0.2, random_state=42):
    """
    Divide datos en train y test
    
    Args:
        df: DataFrame
        test_size: Proporción de test
        random_state: Semilla
    
    Returns:
        tuple: (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test


def get_descriptive_stats(df):
    """
    Calcula estadísticas descriptivas
    
    Args:
        df: DataFrame
    
    Returns:
        DataFrame: Estadísticas descriptivas
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.describe()


def export_csv(df, filename='data.csv'):
    """
    Exporta DataFrame a CSV
    
    Args:
        df: DataFrame
        filename: Nombre del archivo
    
    Returns:
        bytes: Contenido CSV
    """
    return df.to_csv(index=False).encode('utf-8')


def export_excel(df, filename='data.xlsx'):
    """
    Exporta DataFrame a Excel
    
    Args:
        df: DataFrame
        filename: Nombre del archivo
    
    Returns:
        bytes: Contenido Excel
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()
