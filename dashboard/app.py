import streamlit as st
from components.i18n import render_top_bar, setup_localized_navigation, t
from components.theme import apply_unified_theme

st.set_page_config(page_title="Weather MLOps Platform", layout="wide")

pg = setup_localized_navigation()


def render_home_page():
    apply_unified_theme()
    render_top_bar(
        "app_title",
        "app_subtitle",
        ["tag_pipeline", "tag_algorithm", "tag_registry"],
        "status_online",
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">{t('metric_cities')}</div>
            <div class="metric-val val-cyan">14</div>
            <div style="color: #64748b; font-size: 0.8rem;">{t('metric_cities_sub')}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">{t('metric_levels')}</div>
            <div class="metric-val val-pink">6 hPa</div>
            <div style="color: #64748b; font-size: 0.8rem;">{t('metric_levels_sub')}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">{t('metric_algorithm')}</div>
            <div class="metric-val val-green">Spark GBT</div>
            <div style="color: #64748b; font-size: 0.8rem;">{t('metric_algorithm_sub')}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">{t('metric_target')}</div>
            <div class="metric-val" style="color: #a855f7;">Netlify Edge</div>
            <div style="color: #64748b; font-size: 0.8rem;">{t('metric_target_sub')}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    box_style = (
        "background: rgba(30, 41, 59, 0.4); "
        "border: 1px solid rgba(255, 255, 255, 0.08); "
        "border-radius: 16px; padding: 24px;"
    )

    st.markdown(
        f"""
    <div style="{box_style}">
        <h3 style="color: #f8fafc; font-size: 1.2rem; margin-bottom: 8px;">{t('arch_title')}</h3>
        <p style="color: #94a3b8; line-height: 1.7; font-size: 0.95rem;">
            {t('arch_desc')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if pg.title in [t("tab_home"), "Inicio", "Home"]:
    render_home_page()
else:
    pg.run()
