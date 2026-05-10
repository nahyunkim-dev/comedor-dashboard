# ============================================================
    modelo.fit(X_train, y_train)

    # Predicciones

    y_pred = modelo.predict(X_test)

    # ========================================================
    # METRICAS
    # ========================================================

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    col6, col7, col8 = st.columns(3)

    with col6:
        st.metric("MAE", round(mae,2))

    with col7:
        st.metric("RMSE", round(rmse,2))

    with col8:
        st.metric("R²", round(r2,2))

    # ========================================================
    # COMPARACION REAL VS PREDICCION
    # ========================================================

    resultados = pd.DataFrame({
        'Real': y_test.values,
        'Prediccion': y_pred
    })

    fig4 = px.line(
        resultados,
        y=['Real','Prediccion'],
        template='plotly_white',
        title='Valores Reales vs Prediccion'
    )

    fig4.update_layout(title_x=0.5)

    st.plotly_chart(fig4, use_container_width=True)

    st.write(
        """
        El modelo presenta una tendencia cercana a los valores
        reales, lo cual demuestra que puede utilizarse como
        herramienta de apoyo para mejorar la planeacion.
        """
    )

# ============================================================
# TAB 3 — CONCLUSIONES Y RECOMENDACIONES
# ============================================================

with analisis3:

    st.subheader("Hallazgos Clave")

    st.write(
        """
        • Existe una relacion directa entre empleados y comidas servidas.

        • El analisis permitio detectar patrones operativos.

        • El modelo predictivo ayuda a reducir 
