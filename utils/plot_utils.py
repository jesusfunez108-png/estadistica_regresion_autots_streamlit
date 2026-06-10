"""
📊 Utilidades de Gráficos
Visualizaciones para análisis de datos y modelos
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_observed_vs_predicted(y_true, y_pred, title="Observado vs Predicho"):
    """
    Gráfico de valores observados vs predichos
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        title: Título del gráfico
    
    Returns:
        fig: Figura de Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=y_true,
        name='Observado',
        mode='lines+markers',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        y=y_pred,
        name='Predicho',
        mode='lines+markers',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Índice',
        yaxis_title='Valor',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_residuals(residuals, title="Distribución de Residuos"):
    """
    Gráfico de residuos
    
    Args:
        residuals: Array de residuos
        title: Título del gráfico
    
    Returns:
        fig: Figura de Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Histograma de Residuos", "Q-Q Plot")
    )
    
    # Histograma
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Residuos',
            marker_color='rgba(255, 107, 53, 0.7)'
        ),
        row=1, col=1
    )
    
    # Q-Q Plot
    from scipy import stats
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode='markers',
            name='Residuos',
            marker=dict(color='rgba(0, 100, 200, 0.6)')
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=500, title_text=title, showlegend=True)
    
    return fig


def plot_scatter(x, y, title="Relación entre Variables", x_label="X", y_label="Y"):
    """
    Gráfico de dispersión
    
    Args:
        x: Variable X
        y: Variable Y
        title: Título
        x_label: Etiqueta eje X
        y_label: Etiqueta eje Y
    
    Returns:
        fig: Figura de Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=8,
            color='rgba(255, 107, 53, 0.7)',
            line=dict(width=1, color='rgba(255, 107, 53, 1)')
        ),
        name='Datos'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_regression_line(x, y, y_pred, title="Regresión Lineal"):
    """
    Gráfico de dispersión con línea de regresión
    
    Args:
        x: Variable independiente
        y: Variable dependiente
        y_pred: Valores predichos
        title: Título
    
    Returns:
        fig: Figura de Plotly
    """
    fig = go.Figure()
    
    # Ordenar para la línea
    sorted_indices = np.argsort(x)
    x_sorted = np.array(x)[sorted_indices]
    y_pred_sorted = np.array(y_pred)[sorted_indices]
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Datos observados',
        marker=dict(
            size=8,
            color='rgba(100, 150, 255, 0.7)'
        )
    ))
    
    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=y_pred_sorted,
        mode='lines',
        name='Línea de regresión',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Variable Independiente',
        yaxis_title='Variable Dependiente',
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_correlation_heatmap(df, title="Matriz de Correlación"):
    """
    Mapa de calor de correlación
    
    Args:
        df: DataFrame
        title: Título
    
    Returns:
        fig: Figura de Plotly
    """
    corr = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlación")
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        width=600
    )
    
    return fig


def plot_time_series(data, title="Serie Temporal", x_label="Tiempo", y_label="Valor"):
    """
    Gráfico de serie temporal
    
    Args:
        data: Array o Series
        title: Título
        x_label: Etiqueta X
        y_label: Etiqueta Y
    
    Returns:
        fig: Figura de Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=data,
        mode='lines',
        name='Serie',
        line=dict(color='#FF6B35', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_forecast_with_intervals(y_true, y_pred, y_lower=None, y_upper=None, 
                                  title="Pronóstico con Intervalos de Confianza"):
    """
    Gráfico de pronóstico con intervalos
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        y_lower: Límite inferior
        y_upper: Límite superior
        title: Título
    
    Returns:
        fig: Figura de Plotly
    """
    fig = go.Figure()
    
    # Valores reales
    fig.add_trace(go.Scatter(
        y=y_true,
        name='Observado',
        mode='lines+markers',
        line=dict(color='blue', width=2)
    ))
    
    # Predicciones
    fig.add_trace(go.Scatter(
        y=y_pred,
        name='Predicho',
        mode='lines+markers',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Intervalo de confianza
    if y_lower is not None and y_upper is not None:
        fig.add_trace(go.Scatter(
            y=y_upper,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            y=y_lower,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Intervalo de Confianza',
            fillcolor='rgba(0,100,200,0.2)'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Período',
        yaxis_title='Valor',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_distribution(data, title="Distribución", x_label="Valor"):
    """
    Gráfico de distribución
    
    Args:
        data: Array de datos
        title: Título
        x_label: Etiqueta X
    
    Returns:
        fig: Figura de Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=30,
        name='Frecuencia',
        marker_color='rgba(255, 107, 53, 0.7)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title='Frecuencia',
        height=500
    )
    
    return fig
