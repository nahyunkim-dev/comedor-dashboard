# ============================================================

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

st.write("""
El modelo presenta una tendencia cercana a los valores reales,
lo cual demuestra que puede utilizarse como herramienta de apoyo
para mejorar la planeacion del comedor.
""")

# ============================================================
# CONCLUSIONES
# ============================================================

st.markdown("---")

st.header("✅ Conclusiones")

st.write("""
• Existe una relacion directa entre empleados y demanda.

• El analisis permite detectar patrones operativos.

• El modelo predictivo ayuda a reducir incertidumbre.

• Las visualizaciones facilitan la toma de decisiones.

• La empresa puede reducir desperdicio alimentario mediante
una mejor planeacion basada en datos.
""")

# ============================================================
# RECOMENDACIONES
# ============================================================

st.markdown("---")

st.header("📌 Recomendaciones")

st.write("""
1. Actualizar diariamente el dataset.

2. Utilizar el dashboard para monitoreo operativo.

3. Implementar alertas de sobreproduccion.

4. Integrar mas variables al modelo predictivo.

5. Continuar evolucionando el sistema de analitica.
""")

# ============================================================
# FIN DEL DASHBOARD
# ============================================================
