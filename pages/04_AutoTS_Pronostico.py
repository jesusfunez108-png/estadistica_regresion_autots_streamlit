"""
🤖 Página 04: AutoTS - Pronósticos Automáticos
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="04. AutoTS Pronóstico", page_icon="🤖", layout="wide")

st.title("🤖 AutoTS - Pronósticos Automáticos")
st.subheader("Selección automática del mejor modelo de series de tiempo")

if st.session_state.dataset is None:
    st.warning("⚠️ Por favor carga datos primero en la página de inicio")
else:
    numeric_cols = st.session_state.dataset.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 1:
        st.error("❌ Necesitas al menos 1 variable numérica")
    else:
        series_col = st.selectbox("Selecciona la serie temporal", numeric_cols)
        series = st.session_state.dataset[series_col].values
        
        st.info("""
        **¿Qué es AutoTS?**
        - Búsqueda automática del mejor modelo ARIMA
        - Prueba múltiples configuraciones
        - Selecciona según criterios de desempeño
        - Ideal para series sin conocimiento previo
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_steps = st.number_input("Pasos a pronosticar", value=10, min_value=1, max_value=100)
        
        with col2:
            train_ratio = st.slider("Ratio entrenamiento", 0.5, 0.9, 0.8)
        
        with col3:
            max_generations = st.number_input("Generaciones MaxAutoML", value=1, min_value=1, max_value=5)
        
        if st.button("🚀 Ejecutar AutoTS"):
            try:
                with st.spinner("🔍 Buscando el mejor modelo..."):
                    from autots import AutoTS
                    
                    # Dividir datos
                    train_size = int(len(series) * train_ratio)
                    series_train = series[:train_size]
                    
                    # Configurar AutoTS
                    model = AutoTS(
                        forecast_length=int(forecast_steps),
                        frequency='D',
                        model_list='fast',
                        max_generations=int(max_generations),
                        num_validations=1,
                        ensemble='simple'
                    )
                    
                    # Entrenar
                    model.fit(series_train)
                    
                    # Pronósticar
                    prediction = model.predict(forecast_length=int(forecast_steps))
                    
                    st.session_state.forecast = prediction
                    st.success("✅ ¡AutoTS completado!")
                    
                    # Mostrar resultados
                    st.header("📊 Resultados")
                    
                    st.subheader("Mejor Modelo")
                    best_model = model.best_model_name
                    best_params = model.best_model_params
                    
                    st.info(f"**Modelo seleccionado**: {best_model}")
                    
                    # Pronóstico
                    st.subheader("Pronóstico")
                    forecast_values = prediction.forecast.values.flatten()
                    
                    from utils.plot_utils import plot_time_series
                    
                    # Combinar histórico y pronóstico
                    combined = np.concatenate([series[-20:], forecast_values])
                    
                    fig = plot_time_series(combined, title=f"Pronóstico AutoTS - {best_model}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla de pronóstico
                    forecast_df = pd.DataFrame({
                        'Período': range(1, len(forecast_values) + 1),
                        'Pronóstico': forecast_values
                    })
                    st.dataframe(forecast_df, use_container_width=True)
                    
            except ImportError:
                st.error("❌ AutoTS no está instalado. Ejecuta: pip install autots")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        st.success("💡 AutoTS selecciona automáticamente el mejor modelo entre múltiples opciones")
