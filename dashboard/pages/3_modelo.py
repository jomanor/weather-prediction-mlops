import os

import streamlit as st
from components.i18n import render_top_bar, t
from components.theme import apply_unified_theme

st.set_page_config(page_title="Model Audit", layout="wide")

apply_unified_theme()
render_top_bar("model_title", "model_subtitle", ["tag_emerald"], "status_registry")

container_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
browser_url = "http://localhost:5000"

box_style = (
    "background: rgba(30, 41, 59, 0.4); "
    "border: 1px solid rgba(255, 255, 255, 0.08); "
    "border-radius: 16px; padding: 24px;"
)

st.markdown(
    f"""
<div style="{box_style} margin-bottom: 20px;">
    <h3 style="color: #38bdf8; margin-bottom: 12px;">{t('model_active')}</h3>
    <ul style="color: #cbd5e1; line-height: 1.8; font-size: 0.95rem;">
        <li><strong>{t('model_uri')}</strong>: <code>{container_uri}</code></li>
        <li><strong>{t('model_url')}</strong>: <code>{browser_url}</code></li>
        <li><strong>{t('model_name')}</strong>: <code>weather_temperature_1h</code></li>
        <li><strong>{t('model_stage')}</strong>: <code>Production</code></li>
        <li><strong>{t('model_alg')}</strong>: Apache Spark GBTRegressor</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

st.link_button(t("btn_mlflow"), browser_url, type="primary")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    f"""
<div style="{box_style}">
    <h3 style="color: #34d399; margin-bottom: 12px;">{t('features_title')}</h3>
    <ul style="color: #cbd5e1; line-height: 1.8; font-size: 0.95rem;">
        <li><code>specific_humidity</code>: Computed via Bolton 1980 approximation formula.</li>
        <li><code>wind_u</code> / <code>wind_v</code>: Cartesian wind vector decomposition.</li>
        <li><code>cape</code>: Convective Available Potential Energy.</li>
        <li><code>precip_sum_6h</code>: 6-hour rolling window accumulated precipitation sum.</li>
        <li><code>pressure_level_*</code>: Isobaric levels (200, 500, 700, 850, 925, 1000 hPa).</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)
