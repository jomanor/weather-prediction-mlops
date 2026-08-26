import streamlit as st
from components.i18n import render_top_bar, t
from components.mongo_client import fetch_latest_predictions
from components.theme import apply_unified_theme

st.set_page_config(page_title="Historical Logs", layout="wide")

apply_unified_theme()
render_top_bar("hist_title", "hist_subtitle", ["tag_purple"], "status_db")

df_preds = fetch_latest_predictions()

if df_preds.empty:
    st.info(t("hist_empty"))
else:
    st.dataframe(df_preds, use_container_width=True, hide_index=True)
