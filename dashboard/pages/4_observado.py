import streamlit as st
from components.charts import build_atmospheric_dashboard_chart
from components.i18n import render_top_bar, t
from components.mongo_client import fetch_observed_weather
from components.theme import apply_unified_theme

st.set_page_config(page_title="Current Data Analytics", layout="wide")

apply_unified_theme()
render_top_bar("obs_title", "obs_subtitle", ["tag_cyan"], "status_sensor")

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
hours = st.sidebar.slider(t("obs_window"), 6, 168, 48)

obs_df = fetch_observed_weather(selected_city, hours=hours)

if obs_df.empty:
    st.warning(t("obs_empty"))
else:
    fig = build_atmospheric_dashboard_chart(obs_df, selected_city)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(t("obs_table_title"))
    st.dataframe(obs_df, use_container_width=True, hide_index=True)
