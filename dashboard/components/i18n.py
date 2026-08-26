import streamlit as st

I18N = {
    "ES": {
        "nav_title": "Navegación",
        "nav_info": (
            "Selecciona una página en el menú lateral para navegar por las métricas en tiempo "
            "real, histórico, auditoría del modelo y analítica de sensores."
        ),
        "status_online": "EN LÍNEA",
        "status_benchmark": "EVALUACIÓN EN VIVO",
        "status_db": "BASE DE DATOS CONECTADA",
        "status_registry": "REGISTRO ONLINE",
        "status_sensor": "SENSOR ACTIVO",
        # Localized Sidebar Navigation Tabs
        "tab_home": "Inicio",
        "tab_comp": "1. Comparación en Vivo",
        "tab_hist": "2. Histórico de Inferencia",
        "tab_model": "3. Auditoría del Modelo",
        "tab_obs": "4. Analítica de Sensores",
        # app.py
        "app_title": "Plataforma de Predicción Meteorológica & MLOps",
        "app_subtitle": (
            "Transmisión de datos meteorológicos en tiempo real, ingeniería de características "
            "físicas y benchmarking de inferencia."
        ),
        "tag_pipeline": "PIPELINE DISTRIBUIDO",
        "tag_algorithm": "SPARK GBT",
        "tag_registry": "REGISTRO MLFLOW",
        "metric_cities": "Ciudades Monitoreadas",
        "metric_cities_sub": "Península Ibérica",
        "metric_levels": "Niveles Atmosféricos",
        "metric_levels_sub": "200 a 1000 hPa",
        "metric_algorithm": "Algoritmo ML",
        "metric_algorithm_sub": "Gradient Boosted Trees",
        "metric_target": "Despliegue Objetivo",
        "metric_target_sub": "Proxy Cloud Edge",
        "arch_title": "Resumen de Arquitectura",
        "arch_desc": (
            "Sistema end-to-end de Operaciones de Aprendizaje Automático (MLOps). Ingesta continua "
            "en tiempo real mediante <strong>Kafka</strong>, ingeniería de características físicas "
            "en <strong>Apache Spark</strong> (humedad específica Bolton, vectores cartesianos de "
            "viento u/v, CAPE), versionado de modelos en <strong>MLflow Registry</strong> e "
            "inferencia REST vía <strong>FastAPI</strong>."
        ),
        # 1_comparacion.py
        "comp_title": "Benchmarking del Modelo en Tiempo Real",
        "comp_subtitle": (
            "Evaluación comparativa de predicciones Spark GBT frente a observaciones Open-Meteo "
            "y predicciones AEMET."
        ),
        "select_city": "Seleccionar Municipio",
        "select_horizon": "Horizonte de Predicción (Horas)",
        "metric_obs": "Temperatura Observada",
        "metric_obs_sub": "Sensor Open-Meteo",
        "metric_model": "Modelo Spark GBT",
        "metric_model_sub": "Modelo Producción MLflow",
        "metric_aemet": "Predicción AEMET",
        "metric_aemet_sub": "API AEMET OpenData",
        # 2_historico.py
        "hist_title": "Predicciones Históricas & Registro de Auditoría",
        "hist_subtitle": (
            "Base de datos de trazabilidad de inferencias generadas por nodo de ciudad."
        ),
        "hist_empty": (
            "No hay predicciones en la base de datos. Ejecuta `python "
            "spark/spark-jobs/inference.py` para generar predicciones."
        ),
        # 3_modelo.py
        "model_title": "Auditoría de Modelos MLflow & Características",
        "model_subtitle": (
            "Parámetros del modelo en producción y definiciones de variables físicas atmosféricas."
        ),
        "model_active": "Modelo Activo en Producción",
        "model_uri": "URI Inter-Servicio Contenedores",
        "model_url": "URL Web de Navegador",
        "model_name": "Nombre Modelo Registrado",
        "model_stage": "Etapa del Modelo",
        "model_alg": "Algoritmo",
        "btn_mlflow": "Abrir Panel de Control MLflow (http://localhost:5000) ->",
        "features_title": "Variables Físicas Atmosféricas",
        # 4_observado.py
        "obs_title": "Analítica de Sensores Atmosféricos en Tiempo Real",
        "obs_subtitle": (
            "Panel multiparámetro de telemetría derivado de observaciones continuas de Open-Meteo."
        ),
        "obs_window": "Ventana Histórica (Horas)",
        "obs_empty": (
            "No se encontraron observaciones. Ejecuta el script de backfill o verifica el "
            "consumidor Kafka."
        ),
        "obs_table_title": "Registro de Telemetría de Sensores",
        # Subplots charts.py
        "chart_temp": "Temperatura y Sensación Térmica (°C)",
        "chart_humidity": "Humedad Relativa (%)",
        "chart_pressure": "Presión Atmosférica en Superficie (hPa)",
        "chart_wind": "Velocidad del Viento (km/h)",
        "chart_precip": "Precipitación & Lluvia Acumulada (mm)",
        "chart_cloud": "Cobertura Nubosa (%)",
        "chart_dashboard_title": "Panel de Observación Atmosférica — ",
    },
    "EN": {
        "nav_title": "Navigation",
        "nav_info": (
            "Select a page from the sidebar menu to navigate through Live Benchmarking, "
            "Historical Metrics, Model Audit, and Live Analytics."
        ),
        "status_online": "ONLINE",
        "status_benchmark": "LIVE BENCHMARK",
        "status_db": "DATABASE CONNECTED",
        "status_registry": "REGISTRY ONLINE",
        "status_sensor": "SENSOR ACTIVE",
        # Localized Sidebar Navigation Tabs
        "tab_home": "Home",
        "tab_comp": "1. Live Benchmarking",
        "tab_hist": "2. Historical Predictions",
        "tab_model": "3. Model Audit",
        "tab_obs": "4. Sensor Analytics",
        # app.py
        "app_title": "Atmospheric Forecast & MLOps Platform",
        "app_subtitle": (
            "Real-time weather streaming, physics feature engineering, and model "
            "inference benchmarking."
        ),
        "tag_pipeline": "DISTRIBUTED PIPELINE",
        "tag_algorithm": "SPARK GBT",
        "tag_registry": "MLFLOW REGISTRY",
        "metric_cities": "Monitored Cities",
        "metric_cities_sub": "Iberian Peninsula",
        "metric_levels": "Atmospheric Levels",
        "metric_levels_sub": "200 to 1000 hPa",
        "metric_algorithm": "ML Algorithm",
        "metric_algorithm_sub": "Gradient Boosted Trees",
        "metric_target": "Target Deployment",
        "metric_target_sub": "Cloud Edge Proxy",
        "arch_title": "Architecture Summary",
        "arch_desc": (
            "Complete end-to-end Machine Learning Operations system. Continuous real-time "
            "ingestion via <strong>Kafka</strong>, physics-informed atmospheric feature "
            "engineering in <strong>Apache Spark</strong> (Bolton specific humidity, cartesian "
            "wind vectors u/v, CAPE), model versioning in <strong>MLflow Registry</strong>, "
            "and REST inference via <strong>FastAPI</strong>."
        ),
        # 1_comparacion.py
        "comp_title": "Live Model Benchmarking",
        "comp_subtitle": (
            "Comparative evaluation of Spark GBT predictions against Open-Meteo observations "
            "and AEMET official forecasts."
        ),
        "select_city": "Select Municipality",
        "select_horizon": "Forecast Horizon (Hours)",
        "metric_obs": "Observed Temperature",
        "metric_obs_sub": "Open-Meteo Sensor Stream",
        "metric_model": "Spark GBT Model",
        "metric_model_sub": "MLflow Production Model",
        "metric_aemet": "AEMET Official",
        "metric_aemet_sub": "AEMET OpenData API",
        # 2_historico.py
        "hist_title": "Historical Predictions & Audit Logs",
        "hist_subtitle": "Traceability database of generated inferences per city node.",
        "hist_empty": (
            "No predictions found in database. Run `python spark/spark-jobs/inference.py` "
            "to populate predictions."
        ),
        # 3_modelo.py
        "model_title": "MLflow Model Registry & Features Audit",
        "model_subtitle": (
            "Production model parameters and atmospheric physics feature definitions."
        ),
        "model_active": "Active Production Model",
        "model_uri": "Container Inter-Service URI",
        "model_url": "Browser Web UI URL",
        "model_name": "Registered Model Name",
        "model_stage": "Model Stage",
        "model_alg": "Algorithm",
        "btn_mlflow": "Open MLflow Tracking Dashboard (http://localhost:5000) ->",
        "features_title": "Physical Atmospheric Features",
        # 4_observado.py
        "obs_title": "Live Sensor Atmospheric Analytics",
        "obs_subtitle": (
            "Multi-parameter telemetry dashboard derived from continuous Open-Meteo observations."
        ),
        "obs_window": "History Window (Hours)",
        "obs_empty": (
            "No observed weather records found in database. Run backfill script or check Kafka "
            "consumer status."
        ),
        "obs_table_title": "Raw Sensor Telemetry Log",
        # Subplots charts.py
        "chart_temp": "Temperature & Feels Like (°C)",
        "chart_humidity": "Relative Humidity (%)",
        "chart_pressure": "Surface Pressure (hPa)",
        "chart_wind": "Wind Speed (km/h)",
        "chart_precip": "Precipitation & Rain Accumulation (mm)",
        "chart_cloud": "Cloud Cover (%)",
        "chart_dashboard_title": "Current Atmospheric Observation Dashboard — ",
    },
}


def get_language():
    """Retrieve current language selection ('ES' or 'EN')."""
    if "language" not in st.session_state:
        st.session_state["language"] = "ES"
    return st.session_state["language"]


def t(key: str) -> str:
    """Translate key based on session language."""
    lang = st.session_state.get("language", "ES")
    return I18N.get(lang, I18N["ES"]).get(key, key)


def setup_localized_navigation():
    """Configure dynamic localized page navigation without emojis."""
    pg_home = st.Page("app.py", title=t("tab_home"), default=True)
    pg_comp = st.Page("pages/1_comparacion.py", title=t("tab_comp"))
    pg_hist = st.Page("pages/2_historico.py", title=t("tab_hist"))
    pg_model = st.Page("pages/3_modelo.py", title=t("tab_model"))
    pg_obs = st.Page("pages/4_observado.py", title=t("tab_obs"))

    pg = st.navigation([pg_home, pg_comp, pg_hist, pg_model, pg_obs])
    return pg


def render_top_bar(title_key: str, subtitle_key: str, tags: list, status_key: str):
    """Render top header banner with top-right language toggle."""
    if "language" not in st.session_state:
        st.session_state["language"] = "ES"

    col_header, col_lang = st.columns([5, 1])

    with col_lang:
        current_lang = st.session_state["language"]
        selected_lang_name = st.selectbox(
            "Lang",
            ["Español (ES)", "English (EN)"],
            index=0 if current_lang == "ES" else 1,
            key="top_lang_selector",
            label_visibility="collapsed",
        )
        new_lang = "ES" if "Español" in selected_lang_name else "EN"
        if new_lang != current_lang:
            st.session_state["language"] = new_lang
            st.rerun()

    tag_html = "".join([f'<span class="tag tag-cyan">{t(tag)}</span>' for tag in tags])

    st.markdown(
        f"""
    <div class="page-header">
        <div>
            {tag_html}
            <h1 class="page-title">{t(title_key)}</h1>
            <div class="page-subtitle">{t(subtitle_key)}</div>
        </div>
        <div class="status-pill">
            <span class="status-dot"></span> {t(status_key)}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
