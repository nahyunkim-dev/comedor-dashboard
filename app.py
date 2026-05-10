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

h1, h2, h3 {
    color: #0B3C5D;
}

[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #D9E2EC;
    padding: 15px;
    border-radius: 10px;
}

section[data-testid="stSidebar"] {
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITULO PRINCIPAL
# ============================================================

st.title("Dashboard Interactivo — Comedor Empresarial")

st.write("""
Este dashboard fue desarrollado para analizar el comportamiento
del comedor empresarial mediante herramientas de análisis de datos.

El objetivo principal es identificar patrones de consumo,
optimizar la planeación diaria y reducir desperdicios alimentarios.
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

# ============================================================
# CREACION DE VARIABLES NUEVAS
# ============================================================

# Diferencia entre empleados y comidas servidas

df['Diferencia'] = (
    df['Empleados'] - df['Comidas_Servidas']
)

# Clasificacion de demanda

df['Nivel_Demanda'] = np.where(
    df['Comidas_Servidas'] >= 110,
    'Alta',
    'Normal'
)

# Clasificacion operativa

df['Estado_Operacion'] = np.where(
    df['Diferencia'] > 20,
    'Posible Sobreproduccion',
    'Operacion Estable'
)

# ============================================================
# SIDEBAR INTERACTIVO
# ============================================================

st.sidebar.title("Filtros Interactivos")

turno = st.sidebar.multiselect(
    "Seleccionar Turno",
    options=df['Turno'].unique(),
    default=df['Turno'].unique()
)

dia = st.sidebar.multiselect(
    "Seleccionar Día",
    options=df['Dia'].unique(),
    default=df['Dia'].unique()
)

demanda = st.sidebar.multiselect(
    "Nivel de Demanda",
    options=df['Nivel_Demanda'].unique(),
    default=df['Nivel_Demanda'].unique()
)

# ============================================================
# FILTRADO
# ============================================================

df_filtrado = df[
    (df['Turno'].isin(turno)) &
    (df['Dia'].isin(dia)) &
    (df['Nivel_Demanda'].isin(demanda))
]

# ============================================================
# KPIs PRINCIPALES
# ============================================================

st.markdown("---")

st.header("Indicadores Clave del Negocio")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Total Comidas Servidas",
        int(df_filtrado['Comidas_Servidas'].sum())
    )

with col2:

    st.metric(
        "Promedio Diario",
        round(df_filtrado['Comidas_Servidas'].mean(), 2)
    )

with col3:

    st.metric(
        "Promedio Empleados",
        round(df_filtrado['Empleados'].mean(), 2)
    )

with col4:

    st.metric(
        "Días Alta Demanda",
        int(
            (df_filtrado['Nivel_Demanda'] == 'Alta').sum()
        )
    )

with col5:

    st.metric(
        "Promedio Diferencia",
        round(df_filtrado['Diferencia'].mean(), 2)
    )

# ============================================================
# STORYTELLING
# ============================================================

st.markdown("---")

st.header("Contexto del Problema")

st.write("""
La empresa presenta diferencias entre la cantidad de alimentos
preparados y la demanda real diaria dentro del comedor empresarial.

En algunos días existe sobreproducción de alimentos,
mientras que en otros la cantidad preparada resulta insuficiente.

El análisis de datos permite comprender el comportamiento
histórico del consumo y desarrollar herramientas predictivas
para optimizar la planeación operativa.
""")

# ============================================================
# TABS INTERACTIVOS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "Análisis Operativo",
    "Análisis Predictivo",
    "Conclusiones"
])

# ============================================================
# TAB 1 — ANALISIS OPERATIVO
# ============================================================

with tab1:

    st.subheader("Tendencia de Comidas Servidas")

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
    Este gráfico permite visualizar cómo cambia
    la demanda alimentaria a lo largo del tiempo.
    """)

    # ========================================================

    st.subheader("Comidas Servidas por Día")

    fig2 = px.bar(
        df_filtrado,
        x='Dia',
        y='Comidas_Servidas',
        color='Nivel_Demanda',
        template='plotly_white',
        title='Consumo por Día'
    )

    fig2.update_layout(
        title_x=0.5
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.write("""
    La visualización permite identificar
    qué días presentan mayor demanda alimentaria.
    """)

    # ========================================================

    st.subheader("Relación entre Empleados y Comidas")

    fig3 = px.scatter(
        df_filtrado,
        x='Empleados',
        y='Comidas_Servidas',
        color='Estado_Operacion',
        size='Comidas_Servidas',
        hover_data=['Dia'],
        template='plotly_white',
        title='Relación entre Empleados y Consumo'
    )

    fig3.update_layout(
        title_x=0.5
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.write("""
    Existe una relación positiva entre el número
    de empleados y la cantidad de comidas servidas.
    """)

    # ========================================================

    st.subheader("Diferencia entre Empleados y Comidas")

    fig4 = px.bar(
        df_filtrado,
        x='Dia',
        y='Diferencia',
        color='Estado_Operacion',
        template='plotly_white',
        title='Diferencia Operativa'
    )

    fig4.update_layout(
        title_x=0.5
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.write("""
    Este gráfico ayuda a detectar posibles
    problemas de sobreproducción o faltantes.
    """)

# ============================================================
# TAB 2 — ANALISIS PREDICTIVO
# ============================================================

with tab2:

    st.subheader("Modelo Predictivo")

    st.write("""
    Se implementó un modelo de Regresión Lineal
    para estimar la cantidad de comidas servidas
    utilizando el número de empleados como variable principal.
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

    st.subheader("Valores Reales vs Predicción")

    resultados = pd.DataFrame({

        'Real': y_test.values,

        'Predicción': y_pred

    })

    fig5 = px.line(
        resultados,
        y=['Real', 'Predicción'],
        template='plotly_white',
        title='Comparación Real vs Predicción'
    )

    fig5.update_layout(
        title_x=0.5
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    st.write("""
    El modelo presenta una tendencia cercana
    a los valores reales observados.
    """)

# ============================================================
# TAB 3 — CONCLUSIONES
# ============================================================

with tab3:

    st.subheader("Hallazgos Principales")

    st.write("""
    • Existe una relación directa entre empleados
      y comidas servidas.

    • La demanda alimentaria cambia según el día operativo.

    • Algunos días presentan diferencias importantes
      entre empleados y comidas servidas.

    • El modelo predictivo ayuda a mejorar
      la planeación alimentaria.
    """)

    # ========================================================

    st.subheader("Recomendaciones Estratégicas")

    recomendaciones = pd.DataFrame({

        'Recomendación': [

            'Actualizar diariamente el dataset',

            'Monitorear indicadores operativos',

            'Reducir desperdicio alimentario',

            'Implementar análisis predictivo continuo',

            'Fortalecer la toma de decisiones basada en datos'

        ]

    })

    st.table(recomendaciones)

    # ========================================================

    st.subheader("Indicador Operativo")

    promedio = df_filtrado['Diferencia'].mean()

    if promedio < 15:

        st.success(
            "Operación estable y controlada"
        )

    elif promedio < 25:

        st.warning(
            "Atención: variación moderada"
        )

    else:

        st.error(
            "Riesgo operativo: posible sobreproducción"
        )

    # ========================================================

    st.subheader("Conclusión General")

    st.write("""
    El dashboard permite transformar datos operativos
    en información visual clara e interactiva,
    facilitando el monitoreo de KPIs,
    la identificación de patrones
    y el apoyo a la toma de decisiones estratégicas.
    """)
