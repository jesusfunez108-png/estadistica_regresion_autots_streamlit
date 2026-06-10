"""
📊 APLICACIÓN EDUCATIVA DE ANÁLISIS ESTADÍSTICO Y PRONÓSTICOS
Regresión, Series de Tiempo y AutoTS para Ingeniería
"""

import streamlit as st
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="📊 Estadística & Regresión",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session state
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "model" not in st.session_state:
    st.session_state.model = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "forecast" not in st.session_state:
    st.session_state.forecast = None

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF6B35;
    }
    .theory-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("""
<div class="main-header">
    <h1>📊 Análisis Estadístico Educativo</h1>
    <p><strong>Regresión • Series de Tiempo • Pronósticos con AutoTS</strong></p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Navegación principal
st.sidebar.title("🧭 Navegación")
st.sidebar.info("""
    **Bienvenido a la Aplicación Educativa de Estadística**
    
    Esta herramienta te permite:
    - 📈 Analizar datos con regresión lineal y no lineal
    - ⏳ Modelar series temporales
    - 🤖 Realizar pronósticos automáticos
    - 📊 Comparar modelos
    - 📤 Exportar resultados
    
    **Sigue el flujo pedagógico en el menú de páginas.**
""")

# Información general
st.title("🎓 Bienvenida")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📚 ¿Qué es?
    Aplicación educativa para aprender:
    - Regresión lineal y no lineal
    - Análisis de series de tiempo
    - Modelos predictivos
    """)

with col2:
    st.markdown("""
    ### 🎯 ¿Para quién?
    - Estudiantes de ingeniería
    - Profesionales en estadística
    - Analistas de datos
    - Investigadores
    """)

with col3:
    st.markdown("""
    ### 🚀 ¿Cómo usarla?
    1. Sube tus datos (CSV/Excel)
    2. Explora y visualiza
    3. Selecciona modelo
    4. Evalúa resultados
    5. Exporta tu trabajo
    """)

# Sección de datos
st.header("📥 Carga de Datos")

from utils.data_utils import load_file, get_sample_data

# Opción 1: Cargar datos del usuario
uploaded_file = st.file_uploader("Sube un archivo CSV o Excel", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    try:
        df = load_file(uploaded_file)
        st.session_state.dataset = df
        st.success(f"✅ Datos cargados exitosamente: {df.shape[0]} filas × {df.shape[1]} columnas")
    except Exception as e:
        st.error(f"❌ Error al cargar: {e}")

# Opción 2: Usar datos de ejemplo
st.divider()
st.subheader("📊 O usa un dataset de ejemplo:")

sample_options = {
    'Caudales de Ríos': 'Caudales de Ríos',
    'Producción Agrícola': 'Producción Agrícola',
    'Temperatura': 'Temperatura',
    'Regresión Simulada': 'Regresión Simulada'
}

col1, col2 = st.columns([3, 1])

with col1:
    selected_sample = st.selectbox(
        "Selecciona un dataset de ejemplo",
        options=list(sample_options.keys()),
        key="sample_select"
    )

with col2:
    if st.button("📂 Cargar", key="load_sample"):
        df_sample = get_sample_data(sample_options[selected_sample])
        if df_sample is not None:
            st.session_state.dataset = df_sample
            st.success(f"✅ Cargado")

# Mostrar datos cargados
if st.session_state.dataset is not None:
    st.divider()
    st.header("📋 Vista Previa de Datos")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas", st.session_state.dataset.shape[0])
    with col2:
        st.metric("Columnas", st.session_state.dataset.shape[1])
    with col3:
        st.metric("Memoria (MB)", round(st.session_state.dataset.memory_usage(deep=True).sum() / 1024**2, 2))
    
    # Mostrar tabla
    with st.expander("📊 Ver tabla completa"):
        st.dataframe(st.session_state.dataset, use_container_width=True)
    
    # Estadísticas descriptivas
    with st.expander("📈 Estadísticas descriptivas"):
        st.dataframe(st.session_state.dataset.describe(), use_container_width=True)

# Footer
st.divider()
st.markdown("""
---
<div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
    <p>📚 <strong>Aplicación Educativa</strong> | Análisis Estadístico y Pronósticos</p>
    <p>🔬 Desarrollada para estudiantes de Ingeniería y Ciencia de Datos</p>
</div>
""", unsafe_allow_html=True)
