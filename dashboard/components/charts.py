import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_comparison_chart(
    observed_df: pd.DataFrame, predicted_df: pd.DataFrame, aemet_df: pd.DataFrame, city: str
):
    fig = go.Figure()

    COLOR_OBSERVED = "#00f2fe"  # Glowing Electric Cyan
    COLOR_MODEL = "#ff007f"  # Electric Neon Pink/Magenta
    COLOR_AEMET = "#00f5d4"  # Bright Mint Green

    if (
        not observed_df.empty
        and "timestamp" in observed_df.columns
        and "temperature" in observed_df.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=observed_df["timestamp"],
                y=observed_df["temperature"],
                mode="lines+markers",
                name="Observed (Open-Meteo)",
                line=dict(color=COLOR_OBSERVED, width=3.5),
                marker=dict(size=6, symbol="circle", color=COLOR_OBSERVED),
                hovertemplate="<b>Observed</b>: %{y:.1f}°C<br><extra></extra>",
            )
        )

    if (
        not predicted_df.empty
        and "prediction_timestamp" in predicted_df.columns
        and "predicted_temperature" in predicted_df.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=predicted_df["prediction_timestamp"],
                y=predicted_df["predicted_temperature"],
                mode="lines+markers",
                name="Spark GBT Model",
                line=dict(color=COLOR_MODEL, width=3.5, dash="dash"),
                marker=dict(size=7, symbol="diamond", color=COLOR_MODEL),
                hovertemplate="<b>ML Prediction</b>: %{y:.1f}°C<br><extra></extra>",
            )
        )

    if not aemet_df.empty and "timestamp" in aemet_df.columns and "aemet_temp" in aemet_df.columns:
        fig.add_trace(
            go.Scatter(
                x=aemet_df["timestamp"],
                y=aemet_df["aemet_temp"],
                mode="lines+markers",
                name="AEMET Official",
                line=dict(color=COLOR_AEMET, width=2.5, dash="dot"),
                marker=dict(size=5, symbol="square", color=COLOR_AEMET),
                hovertemplate="<b>AEMET Official</b>: %{y:.1f}°C<br><extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>Temperature Forecast & Observation Overlay — {city}</b>",
            font=dict(family="Plus Jakarta Sans", size=20, color="#f8fafc"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(text="Time (UTC)", font=dict(color="#94a3b8", size=13)),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(color="#94a3b8", family="JetBrains Mono"),
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Temperature (°C)", font=dict(color="#94a3b8", size=13)),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(color="#94a3b8", family="JetBrains Mono"),
            zeroline=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            bordercolor="rgba(255, 255, 255, 0.15)",
            font=dict(family="Plus Jakarta Sans", color="#f8fafc", size=13),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1", size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        height=480,
    )
    return fig


def build_atmospheric_dashboard_chart(df: pd.DataFrame, city: str):
    """Build multi-panel atmospheric dashboard for current/historical observed data."""
    if df.empty or "timestamp" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No observation data found for selected period",
            showarrow=False,
            font=dict(color="#94a3b8", size=16),
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)")
        return fig

    from components.i18n import t

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            t("chart_temp"),
            t("chart_humidity"),
            t("chart_pressure"),
            t("chart_wind"),
            t("chart_precip"),
            t("chart_cloud"),
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )

    # Panel 1: Temperature & Dew Point
    if "temperature" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["temperature"],
                name="Temperature",
                line=dict(color="#00f2fe", width=3),
            ),
            row=1,
            col=1,
        )
    if "feels_like" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["feels_like"],
                name="Feels Like",
                line=dict(color="#ff007f", width=2, dash="dot"),
            ),
            row=1,
            col=1,
        )

    # Panel 2: Humidity
    if "humidity" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["humidity"],
                name="Humidity (%)",
                line=dict(color="#00f5d4", width=3),
                fill="tozeroy",
                fillcolor="rgba(0, 245, 212, 0.1)",
            ),
            row=1,
            col=2,
        )

    # Panel 3: Pressure
    if "pressure" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["pressure"],
                name="Surface Pressure",
                line=dict(color="#a855f7", width=3),
            ),
            row=2,
            col=1,
        )

    # Panel 4: Wind Speed
    if "wind_speed" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["wind_speed"],
                name="Wind Speed",
                line=dict(color="#eab308", width=3),
            ),
            row=2,
            col=2,
        )

    # Panel 5: Precipitation & Rain Accumulation (clean filled line chart)
    if "precipitation" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["precipitation"],
                mode="lines",
                name="Precipitation (mm)",
                line=dict(color="#38bdf8", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.12)",
                hovertemplate="<b>Precipitation</b>: %{y:.2f} mm<br><extra></extra>",
            ),
            row=3,
            col=1,
        )
        max_p = df["precipitation"].max() if not df["precipitation"].empty else 1.0
        fig.update_yaxes(range=[-0.05, max(1.0, float(max_p) * 1.25)], row=3, col=1)

    # Panel 6: Cloud Cover
    if "cloud_cover" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["cloud_cover"],
                name="Cloud Cover (%)",
                line=dict(color="#cbd5e1", width=2),
                fill="tozeroy",
                fillcolor="rgba(203, 213, 225, 0.1)",
            ),
            row=3,
            col=2,
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{t('chart_dashboard_title')}{city}</b>",
            font=dict(family="Plus Jakarta Sans", size=20, color="#f8fafc"),
            x=0.01,
            y=0.98,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
        showlegend=False,
        height=800,
        margin=dict(l=30, r=30, t=80, b=30),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")

    return fig
