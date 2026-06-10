# 📊 Aplicación Educativa de Análisis Estadístico y Pronósticos

**Regresión • Series de Tiempo • Pronósticos Automáticos con AutoTS**

Una aplicación web interactiva desarrollada con Streamlit para estudiantes de ingeniería, analistas de datos y profesionales en estadística.

## 🎯 Características Principales

### 📈 Análisis de Regresión
- **Regresión Lineal Simple**: Modelado de relaciones lineales básicas
- **Regresión Lineal Múltiple**: Análisis con múltiples variables predictoras
- **Regresión Polinómica**: Ajuste de curvas no lineales (grados 1-5)
- **Regresión Logística**: Clasificación binaria con probabilidades

### ⏳ Series de Tiempo
- **Descomposición**: Tendencia, estacionalidad y residuos
- **Tests de Estacionariedad**: ADF y KPSS
- **Modelos ARIMA**: Especificación manual de parámetros
- **AutoTS**: Selección automática del mejor modelo

### 📊 Evaluación de Modelos
- **Métricas**: R², RMSE, MAE, MAPE
- **Diagnósticos**: Normalidad, homocedasticidad, linealidad
- **Residuos**: Gráficos y estadísticas
- **Intervalos de confianza**: Predicciones con incertidumbre

### 📤 Exportación
- CSV, Excel y JSON
- Reportes textuales completos
- Datos procesados y predicciones

---

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/jesusfunez108-png/estadistica_regresion_autots_streamlit.git
cd estadistica_regresion_autots_streamlit
```

### 2. Crear ambiente virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 💻 Ejecución

```bash
streamlit run app.py
```

Abre tu navegador en: **http://localhost:8501**

---

## 📚 Estructura de la Aplicación

```
estadistica_streamlit/
│
├── app.py                          # Página principal
│
├── pages/
│   ├── 00_Inicio.py               # Teoría general
│   ├── 01_Regresion_Lineal.py     # Regresión lineal simple
│   ├── 02_No_Lineal_Logistica.py  # Polinómica y logística
│   ├── 03_Series_Tiempo.py        # Series temporales
│   ├── 04_AutoTS_Pronostico.py    # Pronósticos automáticos
│   ├── 05_Auditoria_Modelos.py    # Evaluación y auditoría
│   └── 06_Exportacion_Resultados.py # Descargas
│
├── utils/
│   ├── data_utils.py              # Carga y limpieza
│   ├── regression_utils.py        # Modelos de regresión
│   ├── time_series_utils.py       # Series de tiempo
│   ├── metrics_utils.py           # Métricas de evaluación
│   └── plot_utils.py              # Visualizaciones
│
├── requirements.txt               # Dependencias
├── .streamlit/config.toml        # Configuración
└── README.md                      # Este archivo
```

---

## 🎓 Flujo Pedagógico

### Paso 1: **Inicio - Teoría**
Aprende conceptos fundamentales de regresión y tipos de modelos.

### Paso 2: **Regresión Lineal**
Modelado simple de relaciones lineales entre variables.

### Paso 3: **Regresión Avanzada**
Explora regresión polinómica y logística.

### Paso 4: **Series de Tiempo**
Análisis de datos temporales y modelos ARIMA.

### Paso 5: **AutoTS**
Pronósticos automáticos sin especificar manualmente parámetros.

### Paso 6: **Auditoría**
Evaluación completa de supuestos estadísticos.

### Paso 7: **Exportación**
Descarga tus análisis en múltiples formatos.

---

## 📊 Datos de Ejemplo

La aplicación incluye 4 datasets de ejemplo:

1. **Caudales de Ríos** (365 registros)
   - Caudal, precipitación, temperatura

2. **Producción Agrícola** (14 años)
   - Producción, área, lluvia, temperatura

3. **Temperatura** (365 registros)
   - Máxima, mínima, humedad

4. **Regresión Simulada** (100 puntos)
   - Variables X e Y con relación lineal

---

## 📋 Requisitos

| Librería | Versión | Uso |
|----------|---------|-----|
| Python | 3.8+ | Lenguaje |
| Streamlit | 1.30+ | Web framework |
| Pandas | 2.0+ | Análisis de datos |
| Scikit-learn | Última | Machine Learning |
| Statsmodels | Última | Series de tiempo |
| AutoTS | 0.6.15 | Pronósticos automáticos |
| Plotly | Última | Visualizaciones |

---

## 🔧 Uso Básico

### Cargar datos
```
1. Ve a la página principal
2. Elige: Subir archivo CSV/Excel o usar dataset de ejemplo
3. Visualiza estadísticas descriptivas
```

### Crear modelo
```
1. Ve a la sección de regresión
2. Selecciona variables X (independiente) e Y (dependiente)
3. Click en "Ajustar Modelo"
4. Visualiza resultados y métricas
```

### Pronóstico
```
1. Ve a Series de Tiempo
2. Selecciona el modelo (ARIMA o AutoTS)
3. Configura parámetros
4. Genera pronóstico
```

### Exportar
```
1. Ve a Exportación
2. Selecciona datos a exportar
3. Elige formato (CSV, Excel, JSON)
4. Descarga archivos
```

---

## 📈 Métricas Disponibles

### Regresión
- **R²**: Coeficiente de determinación (0-1)
- **RMSE**: Error cuadrático medio
- **MAE**: Error absoluto medio
- **MAPE**: Error porcentual medio

### Clasificación
- **Exactitud**: Proporción correcta
- **Precisión**: Positivos verdaderos
- **Recall**: Cobertura positivos
- **F1-Score**: Promedio armónico

---

## 🎨 Configuración Personalizada

Edita `.streamlit/config.toml` para cambiar:
- **Colores**: Tema (naranja #FF6B35)
- **Puerto**: Default 8501
- **Fuente**: Sans serif

```toml
[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
font = "sans serif"
```

---

## 🐛 Troubleshooting

### AutoTS no funciona
```bash
pip install autots==0.6.15
```

### Port 8501 ya en uso
```bash
streamlit run app.py --logger.level=debug --server.port 8502
```

### Errores de sesión
```python
# Limpia caché de Streamlit
streamlit cache clear
```

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Predecir producción agrícola
1. Carga dataset "Producción Agrícola"
2. Regresión Lineal: Producción vs Lluvia
3. Análisis de residuos
4. Exporta predicciones

### Ejemplo 2: Pronóstico de caudales
1. Carga "Caudales de Ríos"
2. Series de Tiempo → AutoTS
3. Genera pronóstico 30 días adelante
4. Auditoría de modelos

### Ejemplo 3: Clasificación binaria
1. Datos propios o ejemplo
2. Regresión Logística
3. Evalúa métricas de clasificación
4. Descarga probabilidades

---

## 📚 Referencias Académicas

- Montgomery, D. C., et al. (2012). *Introduction to Linear Regression Analysis*
- Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*
- Hastie, T., Tibshirani, R., & James, G. (2013). *An Introduction to Statistical Learning*

---

## 👨‍💻 Desarrollo

### Stack Tecnológico
- **Frontend**: Streamlit
- **Backend**: Python
- **Datos**: Pandas
- **ML**: Scikit-learn, Statsmodels
- **Visualización**: Plotly

### Características Futuras
- [ ] Modelos XGBoost y Random Forest
- [ ] Análisis de componentes principales (PCA)
- [ ] Validación cruzada avanzada
- [ ] Importación de APIs (FRED, Quandl)
- [ ] Dashboards personalizados

---

## 📄 Licencia

Este proyecto es de código abierto bajo licencia MIT.

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! 

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## ⭐ Agradecimientos

- Streamlit por la fantástica librería
- Scikit-learn y Statsmodels por herramientas robustas
- Comunidad de data science por inspiración

---

**Hecho con ❤️ para estudiantes de ingeniería**

**Última actualización**: Junio 2026

---

## 📋 Checklist de Instalación

- [ ] Clonar repositorio
- [ ] Crear ambiente virtual
- [ ] Instalar `pip install -r requirements.txt`
- [ ] Ejecutar `streamlit run app.py`
- [ ] Abrir http://localhost:8501
- [ ] Cargar datos de ejemplo
- [ ] Probar primera regresión
- [ ] ¡Éxito! 🎉
