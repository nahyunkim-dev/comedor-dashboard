# ============================================================
# DASHBOARD INTERACTIVO — COMEDOR EMPRESARIAL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIGURACION GENERAL
# ============================================================

st.set_page_config(
    page_title="Dashboard Comedor Empresarial",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# TITULO
# ============================================================

st.title("📊 Dashboard Interactivo — Comedor Empresarial")

st.write("""
Este dashboard analiza el comportamiento del comedor empresarial
para identificar patrones de consumo y mejorar la planeación
de alimentos mediante análisis de datos y Machine Learning.
""")

# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data
def cargar_datos():
    df = pd.read_csv("comedor_empresa.csv")
    return df

df = cargar_datos()

# ============================================================
# PREPROCESAMIENTO
# ============================================================

df['Fecha'] = pd.to_datetime(df['Fecha'])

df['Mes'] = df['Fecha'].dt.month

df['Dia_Numero'] = df['Fecha'].dt.day

df['Alta_Demanda'] = np.where(
    df['Comidas_Servidas'] > 100,
    1,
    0
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Filtros")

turno = st.sidebar.multiselect(
    "Selecciona Turno",
    options=df['Turno'].unique(),
    default=df['Turno'].unique()
)

df_filtrado = df[df['Turno'].isin(turno)]

# ============================================================
# KPIs
# ============================================================

st.header("📌 Indicadores Principales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Comidas",
        int(df_filtrado['Comidas_Servidas'].sum())
    )

with col2:
    st.metric(
        "Promedio Diario",
        round(df_filtrado['Comidas_Servidas'].mean(),2)
    )

with col3:
    st.metric(
        "Total Empleados",
        int(df_filtrado['Empleados'].sum())
    )

with col4:
    st.metric(
        "Dias Alta Demanda",
        int(df_filtrado['Alta_Demanda'].sum())
    )

# ============================================================
# GRAFICO 1
# ============================================================

st.markdown("---")

st.header("📈 Tendencia de Consumo")

fig1 = px.line(
    df_filtrado,
    x='Fecha',
    y='Comidas_Servidas',
    title='Comidas Servidas por Fecha'
)

st.plotly_chart(fig1, use_container_width=True)

# ============================================================
# GRAFICO 2
# ============================================================

st.markdown("---")

st.header("📊 Consumo por Dia")

fig2 = px.bar(
    df_filtrado,
    x='Dia',
    y='Comidas_Servidas',
    color='Turno',
    title='Consumo por Dia de la Semana'
)

st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# GRAFICO 3
# ============================================================

st.markdown("---")

st.header("🔍 Relacion Empleados vs Comidas")

fig3 = px.scatter(
    df_filtrado,
    x='Empleados',
    y='Comidas_Servidas',
    color='Turno',
    size='Comidas_Servidas'
)

st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# HEATMAP
# ============================================================

st.markdown("---")

st.header("🔥 Correlacion")

corr = df_filtrado[['Empleados','Comidas_Servidas']].corr()

fig, ax = plt.subplots(figsize=(6,4))

sns.heatmap(
    corr,
    annot=True,
    cmap='Blues',
    ax=ax
)

st.pyplot(fig)

# ============================================================
# MODELO PREDICTIVO
# ============================================================

st.markdown("---")

st.header("🤖 Modelo Predictivo")

X = df_filtrado[['Empleados']]

y = df_filtrado['Comidas_Servidas']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

modelo = LinearRegression()

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

# ============================================================
# METRICAS
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

col5, col6, col7 = st.columns(3)

with col5:
    st.metric("MAE", round(mae,2))

with col6:
    st.metric("RMSE", round(rmse,2))

with col7:
    st.metric("R2", round(r2,2))

# ============================================================
# COMPARACION REAL VS PREDICCION
# ============================================================

st.markdown("---")

st.header("📉 Comparacion Real vs Prediccion")

resultados = pd.DataFrame({
    'Real': y_test.values,
    'Prediccion': y_pred
})

fig4 = px.line(
    resultados,
    y=['Real','Prediccion'],
    title='Valores Reales vs Prediccion'
)

st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# CONCLUSIONES
# ============================================================

st.markdown("---")

st.header("✅ Conclusiones")

st.write("""
• Existe relacion entre empleados y comidas servidas.

• El modelo predictivo ayuda a mejorar la planeacion.

• El dashboard facilita la toma de decisiones.

• El analisis reduce desperdicio alimentario.
""")
