"""
⏳ Página 03: Análisis de Series de Tiempo
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.time_series_utils import (
    decompose_series, moving_average, exponential_smoothing,
    adf_test, arima_parameters, fit_arima, forecast_arima
)
from utils.plot_utils import plot_time_series, plot_forecast_with_intervals

st.set_page_config(page_title="03. Series de Tiempo", page_icon="⏳", layout="wide")

st.title("⏳ Análisis de Series de Tiempo")
st.subheader("Descomposición, tendencias y pronósticos")

if st.session_state.dataset is None:
    st.warning("⚠️ Por favor carga datos primero en la página de inicio")
else:
    numeric_cols = st.session_state.dataset.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 1:
        st.error("❌ Necesitas al menos 1 variable numérica")
    else:
        series_col = st.selectbox("Selecciona la serie temporal", numeric_cols)
        series = st.session_state.dataset[series_col].values
        
        tabs = st.tabs(["Exploración", "Descomposición", "ARIMA", "Pronóstico"])
        
        with tabs[0]:
            st.header("📊 Exploración de la Serie")
            
            # Gráfico de la serie
            fig = plot_time_series(series, title=series_col)
            st.plotly_chart(fig, use_container_width=True)
            
            # Test ADF
            st.subheader("Test de Estacionariedad (ADF)")
            adf_result = adf_test(series)
            
            if 'error' not in adf_result:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Estadístico", f"{adf_result['estadístico']:.4f}")
                with col2:
                    st.metric("P-valor", f"{adf_result['p_valor']:.4f}")
                with col3:
                    status = "✅ Estacionaria" if adf_result['es_estacionaria'] else "❌ No estacionaria"
                    st.metric("Estado", status)
        
        with tabs[1]:
            st.header("📈 Descomposición de la Serie")
            
            period = st.slider("Período estacional", 2, len(series)//2, 12)
            
            if st.button("📊 Descomponer"):
                try:
                    decomp = decompose_series(series, period=period)
                    
                    if 'error' not in decomp:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_trend = plot_time_series(decomp['tendencia'], title="Tendencia")
                            st.plotly_chart(fig_trend, use_container_width=True)
                        
                        with col2:
                            fig_seasonal = plot_time_series(decomp['estacional'], title="Estacionalidad")
                            st.plotly_chart(fig_seasonal, use_container_width=True)
                        
                        fig_residual = plot_time_series(decomp['residual'], title="Residuos")
                        st.plotly_chart(fig_residual, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with tabs[2]:
            st.header("🔧 Modelo ARIMA")
            
            # Sugerir parámetros
            params = arima_parameters(series)
            
            if 'sugerencia' in params:
                st.info(f"**Parámetros sugeridos**: {params['sugerencia']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    p = st.number_input("p (AR)", value=params['p'], min_value=0, max_value=5)
                with col2:
                    d = st.number_input("d (I)", value=params['d'], min_value=0, max_value=2)
                with col3:
                    q = st.number_input("q (MA)", value=params['q'], min_value=0, max_value=5)
                
                if st.button("🔧 Ajustar ARIMA"):
                    try:
                        model = fit_arima(series, order=(int(p), int(d), int(q)))
                        if model:
                            st.session_state.model = model
                            st.success(f"✅ Modelo ARIMA({int(p)},{int(d)},{int(q)}) ajustado")
                        else:
                            st.error("❌ Error al ajustar el modelo")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with tabs[3]:
            st.header("🚀 Pronóstico")
            
            if st.session_state.model is not None:
                steps = st.slider("Pasos a pronosticar", 1, 100, 10)
                
                if st.button("📊 Generar Pronóstico"):
                    try:
                        forecast_result = forecast_arima(st.session_state.model, steps=steps)
                        
                        if 'error' not in forecast_result:
                            y_forecast = forecast_result['pronóstico']
                            conf_int = forecast_result['conf_int']
                            
                            # Combinar serie original con pronóstico
                            y_combined = np.concatenate([series[-20:], y_forecast])
                            y_lower = np.concatenate([np.full(20, np.nan), conf_int[:, 0]])
                            y_upper = np.concatenate([np.full(20, np.nan), conf_int[:, 1]])
                            
                            fig = plot_forecast_with_intervals(y_combined, y_combined, y_lower, y_upper,
                                                              title=f"Pronóstico {steps} pasos adelante")
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Primero ajusta un modelo ARIMA")
