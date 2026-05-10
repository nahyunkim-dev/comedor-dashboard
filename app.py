
# ============================================================
# DASHBOARD INTERACTIVO
# ANALISIS DE DATOS PARA NEGOCIOS
# COMEDOR EMPRESARIAL
# ESTUDIANTE: NAHYUN KIM
# ============================================================

# ============================================================
# IMPORTACION DE LIBRERIAS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# CONFIGURACION GENERAL DEL DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Dashboard Comedor Empresarial",
    layout="wide"
)

# ============================================================
# ESTILO VISUAL
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #F5F7FA;
}

h1, h2, h3 {
    color: #0B3C5D;
}

[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #D9E2EC;
    padding: 15px;
    border-radius: 10px;
}

.sidebar .sidebar-content {
    background-color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITULO PRINCIPAL
# ============================================================

st.title("Dashboard Interactivo — Comedor Empresarial")

st.write("""
Este dashboard fue desarrollado para analizar el comportamiento
operativo del comedor empresarial mediante herramientas de análisis
de datos y modelado predictivo.

El objetivo principal es identificar patrones de consumo,
mejorar la planeación alimentaria y reducir desperdicios
mediante decisiones basadas en datos.
""")

# ============================================================
# CARGA DEL DATASET
# ============================================================

@st.cache_data
def cargar_datos():

    df = pd.read_csv("comedor_empresa.csv")

    return df

df = cargar_datos()

# ============================================================
# PREPROCESAMIENTO DE DATOS
# ============================================================

# Conversión de fecha
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Nuevas variables
df['Mes'] = df['Fecha'].dt.month

df['Dia_Numero'] = df['Fecha'].dt.day

# Variable de alta demanda
df['Alta_Demanda'] = np.where(
    df['Comidas_Servidas'] > 100,
    "Alta",
    "Normal"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Filtros")

# Filtro por turno
turno = st.sidebar.multiselect(
    "Seleccionar Turno",
    options=df['Turno'].unique(),
    default=df['Turno'].unique()
)

# Filtro por día
dia = st.sidebar.multiselect(
    "Seleccionar Día",
    options=df['Dia'].unique(),
    default=df['Dia'].unique()
)

# Aplicación de filtros
df_filtrado = df[
    (df['Turno'].isin(turno)) &
    (df['Dia'].isin(dia))
]

# ============================================================
# SECCION 1
# CONTEXTO GENERAL DEL PROBLEMA
# ============================================================

st.markdown("---")

st.header("Indicadores Generales")

# ============================================================
# KPIs PRINCIPALES
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Comidas",
        int(df_filtrado['Comidas_Servidas'].sum())
    )

with col2:
    st.metric(
        "Promedio Diario",
        round(df_filtrado['Comidas_Servidas'].mean(), 2)
    )

with col3:
    st.metric(
        "Total Empleados",
        int(df_filtrado['Empleados'].sum())
    )

with col4:
    st.metric(
        "Días Alta Demanda",
        int(
            (df_filtrado['Alta_Demanda'] == "Alta").sum()
        )
    )

with col5:
    st.metric(
        "Máximo Consumo",
        int(df_filtrado['Comidas_Servidas'].max())
    )

# ============================================================
# NARRATIVA DE NEGOCIO
# ============================================================

st.markdown("""
### Contexto del Problema

La empresa presenta diferencias entre la cantidad de alimentos
preparados y la demanda real diaria del comedor empresarial.

Esta situación provoca:
- desperdicio alimentario,
- incremento de costos,
- problemas de planeación operativa.

El análisis de datos permite identificar patrones históricos
de consumo y desarrollar modelos predictivos que ayuden
a optimizar la toma de decisiones.
""")

# ============================================================
# SECCION 2
# ANALISIS OPERATIVO
# ============================================================

st.markdown("---")

st.header("Análisis Operativo")

# ============================================================
# GRAFICO DE LINEA
# ============================================================

fig1 = px.line(
    df_filtrado,
    x='Fecha',
    y='Comidas_Servidas',
    title='Comidas Servidas por Fecha',
    template='plotly_white'
)

fig1.update_layout(
    title_x=0.5
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

st.write("""
Este gráfico permite visualizar el comportamiento del consumo
de alimentos a lo largo del tiempo e identificar variaciones
operativas y días de alta demanda.
""")

# ============================================================
# GRAFICO DE BARRAS
# ============================================================

fig2 = px.bar(
    df_filtrado,
    x='Dia',
    y='Comidas_Servidas',
    color='Turno',
    title='Comidas Servidas por Día de la Semana',
    template='plotly_white'
)

fig2.update_layout(
    title_x=0.5
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.write("""
La visualización permite comparar el comportamiento
de consumo según el día de operación y detectar
periodos de mayor demanda alimentaria.
""")

# ============================================================
# SCATTER PLOT
# ============================================================

fig3 = px.scatter(
    df_filtrado,
    x='Empleados',
    y='Comidas_Servidas',
    color='Turno',
    size='Comidas_Servidas',
    title='Relación entre Empleados y Comidas Servidas',
    template='plotly_white'
)

fig3.update_layout(
    title_x=0.5
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.write("""
Se observa una relación positiva entre la cantidad
de empleados presentes y el número de comidas servidas,
lo cual confirma que la asistencia influye directamente
en la demanda alimentaria.
""")

# ============================================================
# SECCION 3
# ANALISIS ESTADISTICO Y PREDICTIVO
# ============================================================

st.markdown("---")

st.header("Análisis Estadístico y Predictivo")

# ============================================================
# HEATMAP DE CORRELACION
# ============================================================

corr = df_filtrado[
    ['Empleados', 'Comidas_Servidas']
].corr()

fig, ax = plt.subplots(figsize=(6,4))

sns.heatmap(
    corr,
    annot=True,
    cmap='Blues',
    ax=ax
)

plt.title("Correlación entre Variables")

st.pyplot(fig)

st.write("""
El análisis correlacional permite identificar
qué variables tienen mayor influencia sobre
la demanda diaria de alimentos.
""")

# ============================================================
# MODELO PREDICTIVO
# ============================================================

# Variables independientes
X = df_filtrado[['Empleados']]

# Variable objetivo
y = df_filtrado['Comidas_Servidas']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Modelo
modelo = LinearRegression()

modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# ============================================================
# METRICAS DEL MODELO
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

st.subheader("Métricas del Modelo Predictivo")

col6, col7, col8 = st.columns(3)

with col6:
    st.metric(
        "MAE",
        round(mae, 2)
    )

with col7:
    st.metric(
        "RMSE",
        round(rmse, 2)
    )

with col8:
    st.metric(
        "R²",
        round(r2, 2)
    )

# ============================================================
# COMPARACION REAL VS PREDICCION
# ============================================================

resultados = pd.DataFrame({
    'Real': y_test.values,
    'Predicción': y_pred
})

fig4 = px.line(
    resultados,
    y=['Real', 'Predicción'],
    title='Valores Reales vs Predicción',
    template='plotly_white'
)

fig4.update_layout(
    title_x=0.5
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

st.write("""
La comparación entre valores reales y predicciones
permite evaluar el desempeño del modelo y validar
su utilidad como herramienta de apoyo para la
planeación operativa.
""")

# ============================================================
# SECCION 4
# RECOMENDACIONES ESTRATEGICAS
# ============================================================

st.markdown("---")

st.header("Conclusiones y Recomendaciones")

st.markdown("""
### Hallazgos Principales

- Existe una relación directa entre empleados y demanda alimentaria.
- El análisis permitió identificar patrones operativos relevantes.
- El modelo predictivo ayuda a reducir incertidumbre en la planeación.
- Las visualizaciones facilitan la interpretación de resultados.

### Recomendaciones Estratégicas

1. Implementar monitoreo continuo mediante dashboards.
2. Actualizar periódicamente el dataset.
3. Integrar nuevas variables al modelo predictivo.
4. Establecer alertas de sobreproducción.
5. Fortalecer la toma de decisiones basada en datos.
""")

# ============================================================
# FIN DEL DASHBOARD
# ============================================================
```
