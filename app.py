import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# CONFIGURACION GENERAL
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

h1 {
    color: #0B3C5D;
}

h2 {
    color: #0B3C5D;
}

[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #D9E2EC;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITULO
# ============================================================

st.title("Dashboard Interactivo — Comedor Empresarial")

st.write("""
Este dashboard permite analizar el comportamiento operativo
del comedor empresarial mediante herramientas de análisis de datos
y visualización interactiva.

El objetivo principal es mejorar la planeación alimentaria,
reducir desperdicios y apoyar la toma de decisiones.
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
# PREPROCESAMIENTO
# ============================================================

df['Fecha'] = pd.to_datetime(df['Fecha'])

df['Mes'] = df['Fecha'].dt.month

df['Nivel_Demanda'] = np.where(
    df['Comidas_Servidas'] > 100,
    'Alta',
    'Normal'
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Filtros Interactivos")

turno = st.sidebar.multiselect(
    "Selecciona Turno",
    options=df['Turno'].unique(),
    default=df['Turno'].unique()
)

dia = st.sidebar.multiselect(
    "Selecciona Día",
    options=df['Dia'].unique(),
    default=df['Dia'].unique()
)

demanda = st.sidebar.multiselect(
    "Nivel de Demanda",
    options=df['Nivel_Demanda'].unique(),
    default=df['Nivel_Demanda'].unique()
)

# ============================================================
# FILTROS
# ============================================================

df_filtrado = df[
    (df['Turno'].isin(turno)) &
    (df['Dia'].isin(dia)) &
    (df['Nivel_Demanda'].isin(demanda))
]

# ============================================================
# KPIs
# ============================================================

st.markdown("---")

st.header("Indicadores Clave del Negocio")

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
        "Máximo Consumo",
        int(df_filtrado['Comidas_Servidas'].max())
    )

with col5:
    st.metric(
        "Días Alta Demanda",
        int(
            (df_filtrado['Nivel_Demanda'] == 'Alta').sum()
        )
    )

# ============================================================
# CONTEXTO DEL PROBLEMA
# ============================================================

st.markdown("---")

st.header("Contexto del Problema")

st.write("""
La empresa presenta diferencias entre la cantidad de alimentos
preparados y la demanda real diaria del comedor empresarial.

Esta situación genera:
- desperdicio alimentario,
- costos innecesarios,
- problemas operativos.

El análisis de datos permite identificar patrones históricos
de consumo y desarrollar modelos predictivos para mejorar
la planeación diaria.
""")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "Análisis Operativo",
    "Análisis Predictivo",
    "Conclusiones"
])

# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.subheader("Tendencia de Consumo")

    fig1 = px.line(
        df_filtrado,
        x='Fecha',
        y='Comidas_Servidas',
        markers=True,
        template='plotly_white',
        title='Comidas Servidas por Fecha'
    )

    fig1.update_layout(title_x=0.5)

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.write("""
    Este gráfico permite identificar tendencias de consumo
    y comportamiento operativo del comedor.
    """)

    # ========================================================

    st.subheader("Consumo por Día")

    fig2 = px.bar(
        df_filtrado,
        x='Dia',
        y='Comidas_Servidas',
        color='Turno',
        template='plotly_white',
        title='Comidas Servidas por Día'
    )

    fig2.update_layout(title_x=0.5)

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.write("""
    Esta visualización ayuda a detectar qué días presentan
    mayor demanda alimentaria.
    """)

    # ========================================================

    st.subheader("Relación entre Empleados y Comidas")

    fig3 = px.scatter(
        df_filtrado,
        x='Empleados',
        y='Comidas_Servidas',
        color='Nivel_Demanda',
        size='Comidas_Servidas',
        hover_data=['Dia', 'Turno'],
        template='plotly_white',
        title='Relación entre Empleados y Comidas Servidas'
    )

    fig3.update_layout(title_x=0.5)

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.write("""
    Existe una relación positiva entre la cantidad de empleados
    presentes y las comidas servidas.
    """)

# ============================================================
# TAB 2
# ============================================================

with tab2:

    st.subheader("Modelo Predictivo")

    st.write("""
    Se implementó un modelo de Regresión Lineal para estimar
    la demanda diaria de alimentos utilizando el número
    de empleados como variable principal.
    """)

    # VARIABLES

    X = df_filtrado[['Empleados']]

    y = df_filtrado['Comidas_Servidas']

    # DIVISION DE DATOS

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # MODELO

    modelo = LinearRegression()

    modelo.fit(X_train, y_train)

    # PREDICCIONES

    y_pred = modelo.predict(X_test)

    # ========================================================

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    st.subheader("Métricas del Modelo")

    col6, col7, col8 = st.columns(3)

    with col6:
        st.metric("MAE", round(mae, 2))

    with col7:
        st.metric("RMSE", round(rmse, 2))

    with col8:
        st.metric("R²", round(r2, 2))

    # ========================================================

    resultados = pd.DataFrame({
        'Real': y_test.values,
        'Predicción': y_pred
    })

    fig4 = px.line(
        resultados,
        y=['Real', 'Predicción'],
        template='plotly_white',
        title='Valores Reales vs Predicción'
    )

    fig4.update_layout(title_x=0.5)

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.write("""
    La comparación entre valores reales y predicciones
    permite evaluar el desempeño del modelo predictivo.
    """)

# ============================================================
# TAB 3
# ============================================================

with tab3:

    st.subheader("Hallazgos Clave")

    st.write("""
    • Existe una relación directa entre empleados y comidas servidas.

    • El análisis permitió detectar patrones operativos relevantes.

    • El modelo predictivo ayuda a mejorar la planeación alimentaria.

    • Las visualizaciones facilitan la interpretación de resultados.
    """)

    # ========================================================

    st.subheader("Recomendaciones Estratégicas")

    recomendaciones = pd.DataFrame({
        'Recomendación': [
            'Actualizar diariamente el dataset',
            'Implementar monitoreo continuo',
            'Integrar nuevas variables al modelo',
            'Reducir sobreproducción alimentaria',
            'Fortalecer la toma de decisiones basada en datos'
        ]
    })

    st.table(recomendaciones)

    # ========================================================

    promedio = df_filtrado['Comidas_Servidas'].mean()

    if promedio > 100:

        st.success(
            'Operación estable — Demanda controlada'
        )

    elif promedio > 70:

        st.warning(
            'Atención — Variación moderada de demanda'
        )

    else:

        st.error(
            'Riesgo operativo — Posibles faltantes'
        )
