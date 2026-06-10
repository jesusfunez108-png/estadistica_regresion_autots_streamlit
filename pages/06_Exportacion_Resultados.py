"""
📤 Página 06: Exportación de Resultados
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

st.set_page_config(page_title="06. Exportación", page_icon="📤", layout="wide")

st.title("📤 Exportación de Resultados")
st.subheader("Descarga tus análisis y modelos")

if st.session_state.dataset is None:
    st.warning("⚠️ Por favor carga datos primero en la página de inicio")
else:
    st.header("📥 Datos Disponibles para Exportar")
    
    # Opciones de exportación
    export_options = st.multiselect(
        "Selecciona qué deseas exportar:",
        [
            "Datos originales",
            "Predicciones",
            "Métricas del modelo",
            "Reporte completo",
            "Visualizaciones"
        ],
        default=["Datos originales", "Predicciones", "Métricas del modelo"]
    )
    
    # Formato de exportación
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox("Formato de exportación", ["CSV", "Excel", "JSON"])
    
    with col2:
        include_index = st.checkbox("Incluir índice", value=False)
    
    if st.button("📥 Generar Archivo", key="export_btn"):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Preparar datos a exportar
            export_data = {}
            
            if "Datos originales" in export_options:
                export_data['datos_originales'] = st.session_state.dataset
            
            if "Predicciones" in export_options and st.session_state.predictions is not None:
                export_data['predicciones'] = pd.DataFrame({
                    'Predicción': st.session_state.predictions
                })
            
            if "Métricas del modelo" in export_options and st.session_state.metrics is not None:
                export_data['metricas'] = pd.DataFrame(
                    st.session_state.metrics,
                    index=[0]
                ).T
            
            # Exportar según formato
            if export_format == "CSV":
                st.info("📁 Descargando múltiples archivos CSV...")
                
                for name, df in export_data.items():
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=include_index)
                    csv_bytes = csv_buffer.getvalue().encode()
                    
                    st.download_button(
                        label=f"⬇️ {name}.csv",
                        data=csv_bytes,
                        file_name=f"{name}_{timestamp}.csv",
                        mime="text/csv",
                        key=f"download_{name}_csv"
                    )
            
            elif export_format == "Excel":
                st.info("📁 Creando archivo Excel...")
                
                excel_buffer = io.BytesIO()
                
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    for name, df in export_data.items():
                        df.to_excel(writer, sheet_name=name[:31], index=include_index)
                
                excel_bytes = excel_buffer.getvalue()
                
                st.download_button(
                    label="⬇️ Descargar Excel",
                    data=excel_bytes,
                    file_name=f"analisis_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
            
            elif export_format == "JSON":
                st.info("📁 Preparando JSON...")
                
                json_data = {}
                for name, df in export_data.items():
                    json_data[name] = df.to_dict(orient='records')
                
                import json
                json_str = json.dumps(json_data, indent=2)
                json_bytes = json_str.encode()
                
                st.download_button(
                    label="⬇️ Descargar JSON",
                    data=json_bytes,
                    file_name=f"analisis_{timestamp}.json",
                    mime="application/json",
                    key="download_json"
                )
            
            st.success("✅ Archivos listos para descargar")
        
        except Exception as e:
            st.error(f"❌ Error al exportar: {str(e)}")
    
    # Reporte de texto
    st.header("📄 Reporte Textual")
    
    if st.button("📋 Generar Reporte Completo"):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            reporte = f"""
════════════════════════════════════════════
REPORTE DE ANÁLISIS ESTADÍSTICO
Generado: {timestamp}
════════════════════════════════════════════

1. DATOS
───────────────
Filas: {st.session_state.dataset.shape[0]}
Columnas: {st.session_state.dataset.shape[1]}
Memoria: {st.session_state.dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB

2. VARIABLES
───────────────
{str(st.session_state.dataset.columns.tolist())}

3. ESTADÍSTICAS DESCRIPTIVAS
───────────────────────────────
{st.session_state.dataset.describe().to_string()}

4. MÉTRICAS DEL MODELO
──────────────────────
{pd.DataFrame(st.session_state.metrics, index=[0]).to_string() if st.session_state.metrics else 'No disponibles'}

════════════════════════════════════════════
"""
            
            st.text_area("Reporte:", value=reporte, height=400)
            
            # Descargar reporte
            reporte_bytes = reporte.encode()
            st.download_button(
                label="⬇️ Descargar Reporte TXT",
                data=reporte_bytes,
                file_name=f"reporte_{timestamp.replace(':', '-').replace(' ', '_')}.txt",
                mime="text/plain",
                key="download_report"
            )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.success("✅ Todos tus análisis pueden ser exportados fácilmente")
