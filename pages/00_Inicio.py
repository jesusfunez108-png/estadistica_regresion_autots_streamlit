"""
📘 Página 00: Inicio - Teoría General de Regresión
Conceptos fundamentales para estudiantes
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="00. Inicio - Teoría", page_icon="📘", layout="wide")

st.title("📘 Conceptos Fundamentales")
st.subheader("Teoría General de Regresión y Estadística")

# Sidebar
st.sidebar.title("📚 Guía de Contenidos")

# Sección 1: ¿Qué es la Regresión?
st.header("1️⃣ ¿Qué es la Regresión?")

st.markdown("""
<div class="theory-box">
    <h3>Definición</h3>
    <p>La <strong>regresión</strong> es una técnica estadística que permite:</p>
    <ul>
        <li><strong>Modelar</strong> la relación entre variables</li>
        <li><strong>Predecir</strong> valores futuros o desconocidos</li>
        <li><strong>Entender</strong> cómo una variable influye en otra</li>
    </ul>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Variable Independiente (X)
    - También llamada **variable explicativa**
    - Es la entrada o predictor
    - Ejemplo: Lluvia, Temperatura, Área de cultivo
    """)

with col2:
    st.markdown("""
    ### 📈 Variable Dependiente (Y)
    - También llamada **variable de respuesta**
    - Es lo que queremos predecir
    - Ejemplo: Producción, Caudal, Ganancia
    """)

# Sección 2: Tipos de Regresión
st.header("2️⃣ Tipos de Regresión")

tabs = st.tabs(["Lineal Simple", "Lineal Múltiple", "Polinómica", "Logística"])

with tabs[0]:
    st.markdown("""
    ### 📍 Regresión Lineal Simple
    
    **Ecuación:** 
    ```
    y = β₀ + β₁·x
    ```
    
    Donde:
    - **β₀**: Intercepción (donde cruza el eje Y)
    - **β₁**: Pendiente (inclinación de la línea)
    - **x**: Variable independiente
    - **y**: Variable dependiente
    
    **Ejemplo:** Predicción de producción agrícola basado en lluvia
    """)
    
    # Gráfico interactivo
    x = np.linspace(0, 10, 100)
    y = 2 + 0.5 * x + np.random.normal(0, 1, 100)
    y_pred = 2 + 0.5 * x
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Datos', marker=dict(color='blue', size=5)))
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name='Línea de regresión', line=dict(color='red', width=3)))
    fig.update_layout(title="Ejemplo de Regresión Lineal Simple", xaxis_title="X", yaxis_title="Y", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.markdown("""
    ### 📍 Regresión Lineal Múltiple
    
    **Ecuación:** 
    ```
    y = β₀ + β₁·x₁ + β₂·x₂ + β₃·x₃ + ...
    ```
    
    Permite usar **múltiples variables predictoras**
    
    **Ejemplo:** Producción = f(Lluvia, Temperatura, Área, Fertilizante)
    
    ✅ **Ventajas:**
    - Modela relaciones complejas
    - Mejor desempeño
    - Analiza múltiples factores
    """)

with tabs[2]:
    st.markdown("""
    ### 📍 Regresión Polinómica
    
    **Ecuación:** 
    ```
    y = β₀ + β₁·x + β₂·x² + β₃·x³ + ...
    ```
    
    Para relaciones **no lineales** (curvas)
    
    **Tipos:**
    - Grado 2: Parábola
    - Grado 3: Curva cúbica
    - Grado n: Polinomio general
    """)
    
    x = np.linspace(-3, 3, 100)
    y = x**2 - 2*x + 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Polinomio', line=dict(color='green', width=3)))
    fig.update_layout(title="Ejemplo de Regresión Polinómica (Grado 2)", xaxis_title="X", yaxis_title="Y", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.markdown("""
    ### 📍 Regresión Logística
    
    **Para clasificación (Sí/No, 0/1)**
    
    **Ecuación:** 
    ```
    P(y=1) = 1 / (1 + e^(-z))
    ```
    
    **Casos de uso:**
    - ¿Lloverá mañana? (Sí/No)
    - ¿Será buena cosecha? (Sí/No)
    - ¿Riesgo de inundación? (Alto/Bajo)
    """)

# Sección 3: Métricas de Evaluación
st.header("3️⃣ Métricas de Evaluación")

st.markdown("""
Para saber si nuestro modelo es **bueno**, usamos métricas:
""")

metric_cols = st.columns(2)

with metric_cols[0]:
    st.markdown("""
    ### 📊 Para Regresión
    
    **R² (Coeficiente de Determinación)**
    - Rango: 0 a 1
    - 1.0 = Modelo perfecto
    - 0.5 = Modelo regular
    - Fórmula: 1 - (SS_res / SS_tot)
    
    **RMSE (Error Cuadrático Medio)**
    - Penaliza errores grandes
    - Menor es mejor
    """)

with metric_cols[1]:
    st.markdown("""
    ### 📊 Para Series de Tiempo
    
    **MAE (Error Absoluto Medio)**
    - Promedio de errores
    - Más interpretable
    
    **MAPE (Error Porcentual)**
    - Porcentaje de error
    - Ideal para comparar series
    """)

# Sección 4: Pasos del Análisis
st.header("4️⃣ Flujo Pedagógico de la Aplicación")

steps = {
    "1. Exploración": "Carga datos y visualiza distribuciones",
    "2. Regresión Lineal": "Modela relaciones lineales simples",
    "3. Regresión Avanzada": "Incluye modelos polinómicos y logística",
    "4. Series de Tiempo": "Analiza datos temporales",
    "5. AutoTS": "Pronósticos automáticos",
    "6. Auditoría": "Compara todos los modelos",
    "7. Exportación": "Descarga tus resultados"
}

for step, description in steps.items():
    st.info(f"**{step}**: {description}")

# Sección 5: Interpretación Académica
st.header("5️⃣ Interpretación Académica")

st.warning("""
⚠️ **IMPORTANTE**: Un modelo estadístico NO es solo números. Debe ser interpretado en el contexto de ingeniería:

1. **¿Tiene sentido el resultado?** ¿Es lógico?
2. **¿Qué dice la teoría?** ¿Concuerda con el dominio?
3. **¿Cuáles son las limitaciones?** ¿Qué supuestos se hacen?
4. **¿Es aplicable?** ¿Sirve para decisiones reales?
""")

# Footer con recomendaciones
st.divider()
st.success("""
✅ **Próximo paso**: Ve a la sección **01. Regresión Lineal** en el menú de páginas
""")
