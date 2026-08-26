import streamlit as st
from components.aemet_client import fetch_aemet_forecast
from components.charts import build_comparison_chart
from components.i18n import render_top_bar, t
from components.mongo_client import fetch_latest_predictions, fetch_observed_weather
from components.theme import apply_unified_theme

st.set_page_config(page_title="Model Benchmarking", layout="wide")

apply_unified_theme()
render_top_bar("comp_title", "comp_subtitle", ["tag_cyan"], "status_benchmark")

CITIES = [
    "Madrid",
    "Barcelona",
    "Valencia",
    "Sevilla",
    "Zaragoza",
    "Malaga",
    "Murcia",
    "Palma",
    "Bilbao",
    "Alicante",
    "Granada",
    "Almería",
    "Paterna",
    "El Ejido",
]

selected_city = st.sidebar.selectbox(t("select_city"), CITIES)
horizon = st.sidebar.slider(t("select_horizon"), 1, 24, 1)

obs_df = fetch_observed_weather(selected_city, hours=24)
pred_df = fetch_latest_predictions(selected_city)
aemet_df = fetch_aemet_forecast(selected_city)

col1, col2, col3 = st.columns(3)

val_obs = (
    f"{obs_df['temperature'].iloc[0]:.1f} °C"
    if not obs_df.empty and "temperature" in obs_df and len(obs_df) > 0
    else "N/A"
)
val_pred = (
    f"{pred_df['predicted_temperature'].iloc[0]:.1f} °C"
    if not pred_df.empty and "predicted_temperature" in pred_df and len(pred_df) > 0
    else "N/A"
)
val_aemet = (
    f"{aemet_df['aemet_temp'].iloc[0]:.1f} °C"
    if not aemet_df.empty and "aemet_temp" in aemet_df and len(aemet_df) > 0
    else "N/A"
)

with col1:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">{t('metric_obs')}</div>
        <div class="metric-val val-cyan">{val_obs}</div>
        <div style="color: #64748b; font-size: 0.8rem;">{t('metric_obs_sub')}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">{t('metric_model')}</div>
        <div class="metric-val val-pink">{val_pred}</div>
        <div style="color: #64748b; font-size: 0.8rem;">{t('metric_model_sub')}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">{t('metric_aemet')}</div>
        <div class="metric-val val-green">{val_aemet}</div>
        <div style="color: #64748b; font-size: 0.8rem;">{t('metric_aemet_sub')}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

fig = build_comparison_chart(obs_df, pred_df, aemet_df, selected_city)
st.plotly_chart(fig, use_container_width=True)
