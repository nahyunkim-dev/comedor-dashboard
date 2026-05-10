import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

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
# ESTILO VISUAL PROFESIONAL
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #F5F7FA;
}

h1 {
    color: #0B3C5D;
    font-size: 40px;
}

h2 {
    color: #0B3C5D;
}

h3 {
    color: #0B3C5D;
}

[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #D9E2EC;
    padding: 15px;
    border-radius: 12px;
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITULO PRINCIPAL
# ============================================================

st.title("Dashboard Interactivo — Comedor Empresarial")

st.write("""
Este dashboard fue desarrollado como parte del proyecto
de Análisis de Datos para Negocios.

El objetivo principal es analizar el comportamiento
del comedor empresarial para identificar patrones de consumo,
mejorar la planeación diaria y reducir desperdicios
mediante herramientas de analítica y visualización interactiva.
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

# Conversion de fecha

df['Fecha'] = pd.to_datetime(df['Fecha'])

# Variables nuevas

df['Mes'] = df['Fecha'].dt.month

df['Dia_Numero'] = df['Fecha'].dt.day

# Clasificacion de demanda

df['Nivel_Demanda'] = np.where(
    df['Comidas_Servidas'] > 100,
    "Alta",
    "Normal"
)

# ============================================================
# SIDEBAR INTERACTIVO
# ============================================================

st.sidebar.title("Filtros Interactivos")

# Filtro de turno

turno = st.sidebar.multiselect(
    "Seleccionar Turno",
    options=df['Turno'].unique(),
    default=df['Turno'].unique()
)

# Filtro de dia

dia = st.sidebar.multiselect(
    "Seleccionar Día",
    options=df['Dia'].unique(),
    default=df['Dia'].unique()
)

# Filtro de nivel de demanda

demanda = st.sidebar.multiselect(
    "Nivel de Demanda",
    options=df['Nivel_Demanda'].unique(),
    default=df['Nivel_Demanda'].unique()
)

# Aplicacion de filtros

df_filtrado = df[
    (df['Turno'].isin(turno)) &
    (df['Dia'].isin(dia)) &
    (df['Nivel_Demanda'].isin(demanda))
]

# ============================================================
# SECCION 1
# INDICADORES CLAVE DEL NEGOCIO
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
            (df_filtrado['Nivel_Demanda'] == "Alta").sum()
        )
    )

# ============================================================
# STORYTELLING
# CONTEXTO DEL PROBLEMA
# ============================================================

st.markdown("---")

st.header("Contexto del Problema")

st.write("""
La empresa presenta diferencias entre la cantidad de alimentos
preparados y la demanda real diaria del comedor empresarial.

Esta situación provoca:
- desperdicio alimentario,
- incremento de costos,
- problemas de planeación operativa.

Por ello se desarrolló un análisis de datos orientado
a identificar patrones históricos de consumo y desarrollar
herramientas predictivas para optimizar la toma de decisiones.
""")

# ============================================================
# TABS INTERACTIVOS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "Análisis Operativo",
    "Análisis Predictivo",
    "Conclusiones y Recomendaciones"
])

# ============================================================
# TAB 1
# ANALISIS OPERATIVO
# ============================================================

with tab1:

    st.subheader("Tendencia de Consumo")

    fig1 = px.line(
        df_filtrado,
        x='Fecha',
        y='Comidas_Servidas',
        markers=True,
        color='Turno',
        template='plotly_white',
        title='Comidas Servidas por Fecha'
    )

    fig1.update_layout(
        title_x=0.5
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.write("""
    Este gráfico permite visualizar el comportamiento
    del consumo alimentario a lo largo del tiempo e identificar
    variaciones operativas y días de mayor demanda.
    """)

    # ========================================================

    st.subheader("Consumo por Día de la Semana")

    fig2 = px.bar(
        df_filtrado,
        x='Dia',
        y='Comidas_Servidas',
        color='Turno',
        barmode='group',
        template='plotly_white',
        title='Comidas Servidas por Día'
    )

    fig2.update_layout(
        title_x=0.5
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.write("""
    Esta visualización permite comparar el comportamiento
    del consumo según el día de operación y detectar
    periodos de mayor demanda alimentaria.
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

    fig3.update_layout(
        title_x=0.5
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.write("""
    El gráfico muestra una relación positiva entre
    la cantidad de empleados y la demanda alimentaria,
    confirmando que la asistencia influye directamente
    en las comidas servidas.
    """)

# ============================================================
# TAB 2
# ANALISIS PREDICTIVO
# ============================================================

with tab2:

    st.subheader("Modelo Predictivo")

    st.write("""
    Se implementó un modelo de Regresión Lineal
    para estimar la demanda diaria de alimentos
    utilizando la cantidad de empleados como variable principal.
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

    # ========================================================

    st.subheader("Comparación Real vs Predicción")

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
    su utilidad como herramienta de apoyo para la planeación.
    """)

# ============================================================
# TAB 3
# CONCLUSIONES Y RECOMENDACIONES
# ============================================================

with tab3:

    st.subheader("Hallazgos Clave")

    st.write("""
    • Existe una relación directa entre empleados
      y comidas servidas.

    • El análisis permitió detectar patrones
      operativos relevantes.

    • El modelo predictivo ayuda a reducir
      incertidumbre en la planeación.

    • Las visualizaciones facilitan la interpretación
      de resultados para usuarios no técnicos.
    """)

    # ========================================================

    st.subheader("Recomendaciones Estratégicas")

    recomendaciones = pd.DataFrame({

        'Recomendación': [

            'Actualizar diariamente el dataset',

            'Implementar monitoreo continuo mediante dashboards',

            'Integrar nuevas variables al modelo predictivo',

            'Reducir sobreproducción alimentaria',

            'Fortalecer la toma de decisiones basada en datos'

        ]

    })

    st.table(recomendaciones)

    # ========================================================

    st.subheader("Indicador Operativo")

    promedio = df_filtrado['Comidas_Servidas'].mean()

    if promedio > 100:

        st.success(
            "Operación estable — Demanda controlada"
        )

    elif promedio > 70:

        st.warning(
            "Atención — Variación moderada de demanda"
        )

    else:

        st.error(
            "Riesgo operativo — Posibles faltantes"
        )

    # ========================================================

    st.subheader("Conclusión General")

    st.write("""
    El dashboard interactivo permite transformar
    datos operativos en información visual clara,
    facilitando el monitoreo de indicadores,
    la identificación de patrones y el apoyo
    a la toma de decisiones estratégicas.
    """)
