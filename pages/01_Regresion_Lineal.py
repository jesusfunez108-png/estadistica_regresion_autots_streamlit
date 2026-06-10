"""
📈 Página 01: Regresión Lineal Simple
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.regression_utils import LinearRegressor
from utils.metrics_utils import calculate_regression_metrics, get_residuals, get_residual_statistics
from utils.plot_utils import plot_scatter, plot_regression_line, plot_residuals

st.set_page_config(page_title="01. Regresión Lineal", page_icon="📈", layout="wide")

st.title("📈 Regresión Lineal Simple")
st.subheader("Modelado de relaciones lineales entre dos variables")

if st.session_state.dataset is None:
    st.warning("⚠️ Por favor carga datos primero en la página de inicio")
else:
    # Seleccionar variables
    numeric_cols = st.session_state.dataset.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("❌ Necesitas al menos 2 variables numéricas")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("Variable Independiente (X)", numeric_cols)
        
        with col2:
            y_col = st.selectbox("Variable Dependiente (Y)", numeric_cols)
        
        if st.button("🔧 Ajustar Modelo", key="fit_linear"):
            try:
                X = st.session_state.dataset[x_col].values
                y = st.session_state.dataset[y_col].values
                
                # Entrenar modelo
                model = LinearRegressor()
                model.fit(X, y)
                st.session_state.model = model
                
                # Predicciones
                y_pred = model.predict(X)
                st.session_state.predictions = y_pred
                
                # Métricas
                metrics = calculate_regression_metrics(y, y_pred, n_features=1)
                st.session_state.metrics = metrics
                
                st.success("✅ Modelo ajustado exitosamente")
                
                # Mostrar resultados
                st.header("📊 Resultados del Modelo")
                
                # Parámetros
                st.subheader("Parámetros del Modelo")
                params = model.get_params()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Intercepción (β₀)", f"{params['Intercepción (β₀)']:.4f}")
                with col2:
                    st.metric("Pendiente (β₁)", f"{params['Pendiente (β₁)']:.4f}")
                
                st.info(f"**Ecuación**: y = {params['Intercepción (β₀)']:.4f} + {params['Pendiente (β₁)']:.4f}·x")
                
                # Métricas
                st.subheader("Métricas de Desempeño")
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                
                with mcol1:
                    st.metric("R²", f"{metrics['R²']:.4f}")
                with mcol2:
                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                with mcol3:
                    st.metric("MAE", f"{metrics['MAE']:.4f}")
                with mcol4:
                    st.metric("MAPE", f"{metrics['MAPE']:.4f}%")
                
                # Gráficos
                st.subheader("Visualizaciones")
                
                tab1, tab2, tab3 = st.tabs(["Modelo", "Residuos", "Estadísticas"])
                
                with tab1:
                    fig = plot_regression_line(X, y, y_pred, title=f"{y_col} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    residuals = get_residuals(y, y_pred)
                    fig = plot_residuals(residuals)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    resid_stats = get_residual_statistics(residuals)
                    st.dataframe(pd.DataFrame(resid_stats, index=[0]).T, use_container_width=True)
                
                # Interpretación
                st.info(f"""
                **Interpretación:**
                - Por cada unidad que aumenta {x_col}, {y_col} aumenta {params['Pendiente (β₁)']:.4f} unidades
                - El modelo explica {metrics['R²']*100:.2f}% de la variabilidad en {y_col}
                """)
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
