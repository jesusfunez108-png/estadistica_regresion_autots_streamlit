"""
🔢 Página 02: Regresión No Lineal y Logística
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.regression_utils import PolynomialRegressor, LogisticRegressor
from utils.metrics_utils import calculate_regression_metrics, calculate_classification_metrics, get_residuals
from utils.plot_utils import plot_regression_line, plot_residuals

st.set_page_config(page_title="02. Regresión Avanzada", page_icon="🔢", layout="wide")

st.title("🔢 Regresión No Lineal y Logística")
st.subheader("Modelos polinómicos y clasificación")

if st.session_state.dataset is None:
    st.warning("⚠️ Por favor carga datos primero en la página de inicio")
else:
    numeric_cols = st.session_state.dataset.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("❌ Necesitas al menos 2 variables numéricas")
    else:
        tab1, tab2 = st.tabs(["Polinómica", "Logística"])
        
        with tab1:
            st.header("📊 Regresión Polinómica")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col = st.selectbox("Variable Independiente (X)", numeric_cols, key="poly_x")
            
            with col2:
                y_col = st.selectbox("Variable Dependiente (Y)", numeric_cols, key="poly_y")
            
            with col3:
                degree = st.slider("Grado del Polinomio", 1, 5, 2)
            
            if st.button("🔧 Ajustar Modelo Polinómico"):
                try:
                    X = st.session_state.dataset[x_col].values
                    y = st.session_state.dataset[y_col].values
                    
                    model = PolynomialRegressor(degree=degree)
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    
                    st.session_state.model = model
                    st.session_state.predictions = y_pred
                    
                    metrics = calculate_regression_metrics(y, y_pred, n_features=degree)
                    st.session_state.metrics = metrics
                    
                    st.success(f"✅ Modelo polinómico de grado {degree} ajustado")
                    
                    # Métricas
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("R²", f"{metrics['R²']:.4f}")
                    with col2:
                        st.metric("R² Ajustado", f"{metrics['R² Ajustado']:.4f}")
                    with col3:
                        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                    with col4:
                        st.metric("MAE", f"{metrics['MAE']:.4f}")
                    
                    # Gráficos
                    tab_plot1, tab_plot2 = st.tabs(["Modelo", "Residuos"])
                    
                    with tab_plot1:
                        fig = plot_regression_line(X, y, y_pred, title=f"Regresión Polinómica (Grado {degree})")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab_plot2:
                        residuals = get_residuals(y, y_pred)
                        fig = plot_residuals(residuals)
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with tab2:
            st.header("🎯 Regresión Logística")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_col_log = st.selectbox("Variable Independiente (X)", numeric_cols, key="log_x")
            
            with col2:
                y_col_log = st.selectbox("Variable Dependiente (Y - Binaria)", numeric_cols, key="log_y")
            
            if st.button("🔧 Ajustar Modelo Logístico"):
                try:
                    X = st.session_state.dataset[x_col_log].values
                    y = st.session_state.dataset[y_col_log].values
                    
                    # Binarizar si es necesario
                    y_binary = (y > y.mean()).astype(int)
                    
                    model = LogisticRegressor()
                    model.fit(X, y_binary)
                    y_pred = model.predict(X)
                    y_proba = model.predict_proba(X)
                    
                    st.session_state.model = model
                    st.session_state.predictions = y_pred
                    
                    metrics = calculate_classification_metrics(y_binary, y_pred)
                    st.session_state.metrics = metrics
                    
                    st.success("✅ Modelo logístico ajustado")
                    
                    # Métricas
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Exactitud", f"{metrics['Exactitud']:.4f}")
                    with col2:
                        st.metric("Precisión", f"{metrics['Precisión']:.4f}")
                    with col3:
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics['F1']:.4f}")
                    
                    st.info("**Interpretación:** El modelo predice la probabilidad de clase 1 vs clase 0")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
