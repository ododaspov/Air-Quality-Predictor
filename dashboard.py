"""
dashboard.py
────────────
Nairobi Real-Time Air Quality Monitoring System
Research-grade Streamlit dashboard with real-world imagery.
Run: streamlit run dashboard.py
"""

import json, os, time, pickle, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Nairobi Air Quality Monitor",
    page_icon="assets/favicon.png" if Path("assets/favicon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
HIST_PATH        = Path(os.getenv("HIST_PATH",        "historical_data.csv"))
MODEL_PATH       = Path(os.getenv("MODEL_PATH",       "air_quality_model.pkl"))
METRICS_PATH     = Path(os.getenv("METRICS_PATH",     "model_metrics.csv"))
FEAT_IMP_PATH    = Path(os.getenv("FEAT_IMP_PATH",    "feature_importances.csv"))
DRIFT_PATH       = Path(os.getenv("DRIFT_PATH",       "drift_report.csv"))
STATUS_PATH      = Path(os.getenv("STATUS_PATH",      "update_status.json"))
RETRAIN_LOG_PATH = Path(os.getenv("RETRAIN_LOG_PATH", "retrain_log.json"))
DATA_URL         = os.getenv("DATA_URL", "")

# Real-world air quality imagery (Unsplash CDN - no API key needed)
HERO_IMAGE    = "https://images.unsplash.com/photo-1611273426858-450d8e3c9fce?w=1400&q=80"  # Nairobi skyline haze
SENSOR_IMAGE  = "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=800&q=80"   # Air quality sensor
CITY_IMAGE    = "https://images.unsplash.com/photo-1569585723035-da1ae4b0ecab?w=800&q=80"   # Urban pollution
WIND_IMAGE    = "https://images.unsplash.com/photo-1466611653911-95081537e5b7?w=800&q=80"   # Wind turbines
FACTORY_IMAGE = "https://images.unsplash.com/photo-1513828583688-c52646db42da?w=800&q=80"   # Industrial air
NATURE_IMAGE  = "https://images.unsplash.com/photo-1448375240586-882707db888b?w=800&q=80"   # Clean air / forest

# ── Scheduler ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _start_scheduler():
    try:
        from auto_update import start_scheduler
        start_scheduler()
        return "ok"
    except Exception as e:
        return str(e)

_start_scheduler()

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography & base ── */
[data-testid="stAppViewContainer"] { background: #080f1a; }
[data-testid="stSidebar"] { background: #0d1526; border-right: 1px solid #1e3048; }

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #0f1f35 0%, #0d1a2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 16px 18px; text-align: center;
    transition: border-color .2s;
}
.kpi-card:hover { border-color: #3b82f6; }
.kpi-label  { font-size: 11px; color: #64748b; letter-spacing: .08em;
               text-transform: uppercase; margin-bottom: 6px; }
.kpi-value  { font-size: 28px; font-weight: 700; color: #f1f5f9; line-height: 1.1; }
.kpi-unit   { font-size: 12px; color: #475569; margin-left: 2px; }
.kpi-sub    { font-size: 11px; color: #475569; margin-top: 5px; }

/* ── Alert banners ── */
.alert-good     { background:#052e16; border-left:3px solid #22c55e;
                   border-radius:6px; padding:10px 16px; color:#86efac; }
.alert-moderate { background:#1c1400; border-left:3px solid #eab308;
                   border-radius:6px; padding:10px 16px; color:#fde047; }
.alert-warning  { background:#1f0f00; border-left:3px solid #f97316;
                   border-radius:6px; padding:10px 16px; color:#fdba74; }
.alert-danger   { background:#1f0000; border-left:3px solid #ef4444;
                   border-radius:6px; padding:10px 16px; color:#fca5a5; }

/* ── Section headers ── */
.sec-hdr {
    font-size: 13px; font-weight: 600; letter-spacing: .06em;
    text-transform: uppercase; color: #64748b;
    margin: 24px 0 12px; padding-bottom: 8px;
    border-bottom: 1px solid #1e3048;
}

/* ── Image caption strip ── */
.img-caption {
    font-size: 11px; color: #475569; text-align: center;
    margin-top: 4px; font-style: italic;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(90deg,#080f1a 0%,rgba(8,15,26,.6) 100%);
    border-radius: 10px; overflow: hidden; position: relative;
    margin-bottom: 24px;
}
.hero-text {
    position: relative; z-index: 2; padding: 28px 32px;
}
.hero-title {
    font-size: 26px; font-weight: 700; color: #f1f5f9;
    line-height: 1.2; margin-bottom: 6px;
}
.hero-sub { font-size: 13px; color: #94a3b8; }

/* ── Pipeline state badge ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 99px;
    font-size: 11px; font-weight: 600; letter-spacing: .04em;
}
.badge-ready      { background:#052e16; color:#86efac; }
.badge-training   { background:#1c1400; color:#fde047; }
.badge-error      { background:#1f0000; color:#fca5a5; }
.badge-idle       { background:#0f1f35; color:#94a3b8; }
.badge-init       { background:#0f1f35; color:#94a3b8; }
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data() -> pd.DataFrame:
    if not HIST_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(HIST_PATH, low_memory=False)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["pm25","pm10","pm1","temperature","humidity","lat","lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_status() -> dict:
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text())
        except Exception:
            pass
    return {"state": "initialising", "message": "Starting up..."}


def who_category(pm25) -> tuple[str, str]:
    """Returns (label, css_class)"""
    try:
        v = float(pm25)
    except (TypeError, ValueError):
        return "Unknown", "alert-good"
    if v <= 12:    return "Good",                           "alert-good"
    if v <= 35.4:  return "Moderate",                       "alert-moderate"
    if v <= 55.4:  return "Unhealthy for Sensitive Groups", "alert-warning"
    if v <= 150.4: return "Unhealthy",                      "alert-danger"
    if v <= 250.4: return "Very Unhealthy",                 "alert-danger"
    return "Hazardous", "alert-danger"


def aqi_color(pm25) -> str:
    try:
        v = float(pm25)
    except (TypeError, ValueError):
        return "#64748b"
    if v <= 12:    return "#22c55e"
    if v <= 35.4:  return "#eab308"
    if v <= 55.4:  return "#f97316"
    if v <= 150.4: return "#ef4444"
    return "#8b5cf6"


def fv(v, d=1):
    try:
        return round(float(v), d)
    except (TypeError, ValueError):
        return None


def safe_mean(s):
    v = s.dropna()
    return round(float(v.mean()), 2) if len(v) else None


CHART_LAYOUT = dict(
    plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font_color="#94a3b8",
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(gridcolor="#1e3048", showgrid=True),
    yaxis=dict(gridcolor="#1e3048", showgrid=True),
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(status: dict, df: pd.DataFrame):
    state = status.get("state", "initialising")
    badge_cls = {
        "ready": "badge-ready", "training": "badge-training",
        "downloading": "badge-training", "checking": "badge-idle",
        "error": "badge-error", "idle": "badge-idle",
    }.get(state, "badge-init")

    st.sidebar.markdown(
        f'<div style="padding:12px 0 8px">'
        f'<div style="font-size:13px;color:#64748b;margin-bottom:6px;font-weight:600;letter-spacing:.06em;text-transform:uppercase">Pipeline</div>'
        f'<span class="badge {badge_cls}">{state.upper()}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.sidebar.caption(status.get("message", ""))
    if status.get("updated_at"):
        st.sidebar.caption(f"Last check: {status['updated_at'][:19]}")
    if status.get("retrained"):
        st.sidebar.success(
            f"Retrained  MAE={status.get('test_mae','?')} ug/m3  "
            f"R2={status.get('test_r2','?')}"
        )

    st.sidebar.divider()
    st.sidebar.markdown(
        '<div style="font-size:13px;color:#64748b;font-weight:600;letter-spacing:.06em;'
        'text-transform:uppercase;margin-bottom:8px">Dataset</div>',
        unsafe_allow_html=True
    )
    if not df.empty:
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Records", f"{len(df):,}")
        if "location" in df.columns:
            c2.metric("Locations", df["location"].nunique())
        if "timestamp" in df.columns:
            st.sidebar.caption(
                f"Period: {df['timestamp'].min().strftime('%d %b')} "
                f"to {df['timestamp'].max().strftime('%d %b %Y')}"
            )
        if "pm25_spike" in df.columns:
            st.sidebar.metric("Spikes detected", int(df["pm25_spike"].sum()))

    st.sidebar.divider()
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                art = pickle.load(f)
            meta = art.get("meta", {})
            tm   = meta.get("test_metrics", {})
            st.sidebar.markdown(
                '<div style="font-size:13px;color:#64748b;font-weight:600;letter-spacing:.06em;'
                'text-transform:uppercase;margin-bottom:8px">ML Model</div>',
                unsafe_allow_html=True
            )
            c1, c2 = st.sidebar.columns(2)
            c1.metric("Type",  meta.get("model_type", "RF").upper())
            c2.metric("Train n", f"{meta.get('n_train', 0):,}")
            c1.metric("MAE",  f"{tm.get('mae', '?')} ug/m3")
            c2.metric("R2",   tm.get("r2", "?"))
            st.sidebar.caption(f"Trained: {str(meta.get('trained_at',''))[:19]}")
        except Exception:
            pass

    st.sidebar.divider()
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Force refresh", use_container_width=True):
        with st.spinner("Running pipeline..."):
            try:
                from auto_update import run_update_cycle
                run_update_cycle(force=True)
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.sidebar.error(str(e))
    if c2.button("Clear cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.checkbox("Auto-refresh every 60s", value=True):
        time.sleep(0.5)
        st.rerun()

    if DATA_URL:
        st.sidebar.divider()
        st.sidebar.caption(f"Source: OpenAfrica sensorsAfrica\n{DATA_URL[:60]}...")


# ── Tab 1: Live Overview ──────────────────────────────────────────────────────
def render_overview(df: pd.DataFrame):
    # Hero image strip
    col_hero, col_info = st.columns([3, 1])
    with col_hero:
        st.image(HERO_IMAGE, use_container_width=True)
        st.markdown(
            '<div class="img-caption">Nairobi city skyline — urban air quality monitoring '
            'captures pollution from traffic, industry and biomass burning</div>',
            unsafe_allow_html=True
        )
    with col_info:
        st.image(SENSOR_IMAGE, use_container_width=True)
        st.markdown(
            '<div class="img-caption">Low-cost particulate sensor (pms5003) '
            'deployed across Nairobi by sensorsAfrica</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sec-hdr">Current Air Quality — Key Indicators</div>', unsafe_allow_html=True)

    if df.empty or "pm25" not in df.columns:
        st.warning("No data loaded yet. Check the pipeline status in the sidebar.")
        return

    latest = df.dropna(subset=["pm25"]).iloc[-1]
    pm25   = fv(latest.get("pm25"))
    pm10   = fv(latest.get("pm10"))
    temp   = fv(latest.get("temperature"))
    hum    = fv(latest.get("humidity"))
    pm1    = fv(latest.get("pm1"))

    cutoff = df["timestamp"].max() - pd.Timedelta(hours=24)
    avg24  = safe_mean(df[df["timestamp"] >= cutoff]["pm25"])
    cat, css = who_category(pm25)

    # Alert banner
    ts_str = latest["timestamp"].strftime("%d %b %Y %H:%M UTC") if "timestamp" in latest.index else ""
    st.markdown(
        f'<div class="{css}" style="margin-bottom:16px">'
        f'<strong>AQI Status: {cat}</strong> &nbsp;|&nbsp; '
        f'PM2.5 = {pm25} ug/m3 &nbsp;|&nbsp; '
        f'24h average = {avg24} ug/m3 &nbsp;|&nbsp; {ts_str}'
        f'</div>',
        unsafe_allow_html=True
    )

    # KPI row
    cols = st.columns(6)
    kpis = [
        ("PM2.5",         pm25,  "ug/m3", "Fine particulate"),
        ("PM10",          pm10,  "ug/m3", "Coarse particulate"),
        ("PM1",           pm1,   "ug/m3", "Ultra-fine"),
        ("Temperature",   temp,  "degC",  "Ambient"),
        ("Humidity",      hum,   "%",     "Relative"),
        ("24h Avg PM2.5", avg24, "ug/m3", "Rolling mean"),
    ]
    for col, (label, val, unit, sub) in zip(cols, kpis):
        v_str = str(val) if val is not None else "--"
        col_color = aqi_color(val) if label in ("PM2.5","24h Avg PM2.5") and val else "#f1f5f9"
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{col_color}">{v_str}'
            f'<span class="kpi-unit">{unit}</span></div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── 1-hour forecast
    st.markdown('<div class="sec-hdr">1-Hour PM2.5 Forecast</div>', unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.info("Model not yet trained. Run: python model_training.py")
    else:
        try:
            from data_pipeline import engineer_features
            from model_training import predict_next_hour

            feat_df = engineer_features(df)
            row     = feat_df.dropna(subset=["pm25"]).iloc[-1]
            result  = predict_next_hour(row)

            p25   = result["pm25_predicted"]
            ci_lo = result["ci_low"]
            ci_hi = result["ci_high"]
            p_cat, p_css = who_category(p25)
            delta = (p25 - pm25) if pm25 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predicted PM2.5 (+1h)", f"{p25:.1f} ug/m3",
                      delta=f"{delta:+.1f} from now")
            c2.metric("90% Confidence Interval", f"{ci_lo:.1f} - {ci_hi:.1f} ug/m3")
            c3.metric("Forecast Category", p_cat)
            c4.metric("Current PM2.5", f"{pm25} ug/m3" if pm25 else "--")

            if p25 > 55.4:
                st.markdown(
                    f'<div class="alert-danger"><strong>ALERT:</strong> Dangerous PM2.5 levels forecast '
                    f'in 1 hour ({p25:.1f} ug/m3). Limit outdoor exposure. '
                    f'Sensitive groups should remain indoors.</div>',
                    unsafe_allow_html=True
                )
            elif p25 > 35.4:
                st.markdown(
                    f'<div class="alert-warning"><strong>CAUTION:</strong> Elevated PM2.5 forecast '
                    f'({p25:.1f} ug/m3). Sensitive groups should reduce prolonged outdoor activity.</div>',
                    unsafe_allow_html=True
                )

            # ── 6-hour forecast chart
            st.markdown('<div class="sec-hdr">6-Hour Multi-Step Forecast</div>', unsafe_allow_html=True)

            lc1, lc2 = st.columns([2, 1])
            with lc2:
                st.image(WIND_IMAGE, use_container_width=True)
                st.markdown(
                    '<div class="img-caption">Wind patterns and meteorological conditions '
                    'significantly influence PM2.5 dispersion in Nairobi</div>',
                    unsafe_allow_html=True
                )
            with lc1:
                cur = feat_df.dropna(subset=["pm25"]).iloc[-1].copy()
                forecasts = []
                for step in range(1, 7):
                    r = predict_next_hour(cur)
                    forecasts.append({
                        "Hour": f"+{step}h",
                        "PM2.5": r["pm25_predicted"],
                        "CI Low": r["ci_low"],
                        "CI High": r["ci_high"],
                    })
                    for lag in range(6, 1, -1):
                        key = f"pm25_lag{lag}"
                        if key in cur.index:
                            cur[key] = cur.get(f"pm25_lag{lag-1}", cur["pm25"])
                    cur["pm25_lag1"] = r["pm25_predicted"]
                    cur["pm25"]      = r["pm25_predicted"]

                fdf = pd.DataFrame(forecasts)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fdf["Hour"], y=fdf["CI High"],
                    fill=None, mode="lines", line_color="rgba(0,0,0,0)", showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=fdf["Hour"], y=fdf["CI Low"],
                    fill="tonexty", fillcolor="rgba(59,130,246,0.12)",
                    mode="lines", line_color="rgba(0,0,0,0)", name="90% CI"
                ))
                fig.add_trace(go.Scatter(
                    x=fdf["Hour"], y=fdf["PM2.5"],
                    mode="lines+markers", name="Forecast",
                    line=dict(color="#3b82f6", width=2.5),
                    marker=dict(size=10, color=[aqi_color(v) for v in fdf["PM2.5"]],
                                line=dict(color="#0a1628", width=1.5))
                ))
                fig.add_hline(y=12,   line=dict(color="#22c55e", dash="dash", width=1),
                              annotation_text="Good (12)")
                fig.add_hline(y=35.4, line=dict(color="#eab308", dash="dash", width=1),
                              annotation_text="Moderate (35.4)")
                fig.add_hline(y=55.4, line=dict(color="#ef4444", dash="dash", width=1),
                              annotation_text="Unhealthy (55.4)")
                fig.update_layout(height=280, yaxis_title="PM2.5 (ug/m3)", **CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Forecast error: {e}")
            with st.expander("Debug info"):
                st.code(traceback.format_exc())

    # ── Pollution trend
    st.markdown('<div class="sec-hdr">Pollution Trend — Last 500 Readings</div>', unsafe_allow_html=True)
    _render_trend(df)

    # ── Diurnal pattern
    st.markdown('<div class="sec-hdr">Diurnal PM2.5 Pattern (Hourly Average)</div>', unsafe_allow_html=True)
    _render_diurnal(df)


def _render_trend(df: pd.DataFrame):
    plot_df = df.dropna(subset=["pm25"]).tail(500).copy()
    if plot_df.empty:
        st.info("No PM2.5 data to plot.")
        return

    fig = go.Figure()
    colors = ["#3b82f6", "#22c55e", "#f97316", "#8b5cf6", "#f43f5e"]

    if "location" in plot_df.columns:
        locs = sorted(plot_df["location"].unique())
        for i, loc in enumerate(locs):
            sub = plot_df[plot_df["location"] == loc]
            fig.add_trace(go.Scatter(
                x=sub["timestamp"], y=sub["pm25"],
                name=f"Sensor Location {loc}",
                mode="lines", line=dict(width=1.8, color=colors[i % len(colors)]),
                opacity=0.9,
            ))
    else:
        fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["pm25"],
                                 name="PM2.5", line=dict(color="#3b82f6", width=2)))

    if "pm10" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df["timestamp"], y=plot_df["pm10"],
            name="PM10 (all locations)", mode="lines",
            line=dict(color="#f97316", width=1, dash="dot"), opacity=0.45
        ))

    fig.add_hline(y=15,   line=dict(color="#22c55e", dash="dash", width=1),
                  annotation_text="WHO 24h guideline (15 ug/m3)",
                  annotation_font=dict(color="#22c55e", size=10))
    fig.add_hline(y=35.4, line=dict(color="#eab308", dash="dash", width=1),
                  annotation_text="Moderate (35.4)",
                  annotation_font=dict(color="#eab308", size=10))
    fig.add_hline(y=55.4, line=dict(color="#ef4444", dash="dash", width=1),
                  annotation_text="Unhealthy (55.4)",
                  annotation_font=dict(color="#ef4444", size=10))

    if "pm25_spike" in plot_df.columns:
        spk = plot_df[plot_df["pm25_spike"]]
        if not spk.empty:
            fig.add_trace(go.Scatter(
                x=spk["timestamp"], y=spk["pm25"],
                mode="markers", name="Anomaly spike",
                marker=dict(color="#ef4444", size=10, symbol="x-open",
                            line=dict(color="#ef4444", width=2))
            ))

    fig.update_layout(
        height=360, yaxis_title="Concentration (ug/m3)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        **CHART_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_diurnal(df: pd.DataFrame):
    if "timestamp" not in df.columns or "pm25" not in df.columns:
        return

    df2 = df.dropna(subset=["pm25"]).copy()
    df2["hour"] = df2["timestamp"].dt.hour
    hourly = df2.groupby("hour")["pm25"].agg(["mean", "std", "count"]).reset_index()
    hourly["std"] = hourly["std"].fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["mean"] + hourly["std"],
        fill=None, mode="lines", line_color="rgba(0,0,0,0)", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["mean"] - hourly["std"],
        fill="tonexty", fillcolor="rgba(59,130,246,0.10)",
        mode="lines", line_color="rgba(0,0,0,0)", name="+/- 1 SD"
    ))
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["mean"],
        mode="lines+markers", name="Hourly mean PM2.5",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=6, color="#3b82f6")
    ))
    fig.add_hline(y=35.4, line=dict(color="#eab308", dash="dash", width=1),
                  annotation_text="Moderate threshold")
    fig.add_hline(y=15,   line=dict(color="#22c55e", dash="dash", width=1),
                  annotation_text="WHO 24h guideline")
    fig.update_layout(
        height=280,
        xaxis=dict(title="Hour of day (UTC+3 local approx.)",
                   tickmode="linear", dtick=2, gridcolor="#1e3048"),
        yaxis=dict(title="PM2.5 (ug/m3)", gridcolor="#1e3048"),
        **CHART_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Sensor Map ─────────────────────────────────────────────────────────
def render_map(df: pd.DataFrame):
    st.markdown('<div class="sec-hdr">Sensor Deployment Map — Nairobi</div>', unsafe_allow_html=True)

    # Context image
    ic1, ic2, ic3 = st.columns(3)
    ic1.image(CITY_IMAGE, use_container_width=True)
    ic1.markdown('<div class="img-caption">Urban pollution sources in Nairobi — '
                 'traffic, cookstoves and industrial activity</div>', unsafe_allow_html=True)
    ic2.image(SENSOR_IMAGE, use_container_width=True)
    ic2.markdown('<div class="img-caption">sensorsAfrica low-cost sensor network '
                 'covering residential and commercial zones</div>', unsafe_allow_html=True)
    ic3.image(NATURE_IMAGE, use_container_width=True)
    ic3.markdown('<div class="img-caption">Clean-air baseline — forested areas surrounding '
                 'Nairobi serve as natural reference zones</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Live Sensor Positions</div>', unsafe_allow_html=True)

    if "lat" not in df.columns or "lon" not in df.columns:
        st.info("No lat/lon data available for map display.")
        return

    latest = (
        df.dropna(subset=["pm25", "lat", "lon"])
          .sort_values("timestamp")
          .groupby("location")
          .last()
          .reset_index()
    )
    if latest.empty:
        st.info("No location data available.")
        return

    latest["AQI Category"] = latest["pm25"].apply(lambda v: who_category(v)[0])
    latest["Bubble Size"]  = latest["pm25"].clip(8, 200)
    latest["Hover Label"]  = latest.apply(
        lambda r: (
            f"<b>Location {r['location']}</b><br>"
            f"PM2.5: {fv(r.get('pm25'))} ug/m3<br>"
            f"PM10: {fv(r.get('pm10'))} ug/m3<br>"
            f"Temperature: {fv(r.get('temperature'))} degC<br>"
            f"Humidity: {fv(r.get('humidity'))} %<br>"
            f"Category: <b>{r['AQI Category']}</b>"
        ), axis=1
    )

    fig = px.scatter_mapbox(
        latest, lat="lat", lon="lon",
        size="Bubble Size", color="pm25",
        color_continuous_scale=[
            [0.0,   "#22c55e"],
            [0.12,  "#22c55e"],
            [0.35,  "#eab308"],
            [0.55,  "#f97316"],
            [0.75,  "#ef4444"],
            [1.0,   "#8b5cf6"],
        ],
        range_color=[0, 100],
        hover_name="Hover Label",
        hover_data={"lat": False, "lon": False, "Bubble Size": False,
                    "pm25": ":.1f", "AQI Category": True},
        zoom=10, height=480, size_max=45,
    )
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="PM2.5<br>(ug/m3)",
            tickfont=dict(color="#94a3b8"),
            titlefont=dict(color="#94a3b8"),
            thickness=12,
        ),
        paper_bgcolor="#0a1628",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Location summary table
    st.markdown('<div class="sec-hdr">Location Summary Table</div>', unsafe_allow_html=True)
    show_cols = [c for c in
                 ["location","lat","lon","pm25","pm10","pm1",
                  "temperature","humidity","AQI Category"] if c in latest.columns]
    display = latest[show_cols].rename(columns={
        "location": "Location", "lat": "Latitude", "lon": "Longitude",
        "pm25": "PM2.5 (ug/m3)", "pm10": "PM10 (ug/m3)", "pm1": "PM1 (ug/m3)",
        "temperature": "Temp (degC)", "humidity": "Humidity (%)",
    }).round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)


# ── Tab 3: Research & Analysis ────────────────────────────────────────────────
def render_research(df: pd.DataFrame):
    st.markdown('<div class="sec-hdr">Research Components — Academic Analysis</div>',
                unsafe_allow_html=True)

    # Context image
    ri1, ri2 = st.columns([1, 3])
    with ri1:
        st.image(FACTORY_IMAGE, use_container_width=True)
        st.markdown(
            '<div class="img-caption">Industrial emission sources contribute to '
            'PM2.5 load — captured by the drift detection module</div>',
            unsafe_allow_html=True
        )
    with ri2:
        st.markdown("""
        This research pipeline addresses four core questions in real-time urban air quality science:

        1. **Predictability** — Can we forecast PM2.5 one hour ahead using lag features, 
           rolling statistics, temperature and humidity? (Random Forest, TimeSeriesSplit CV)

        2. **Anomaly detection** — Can we automatically flag abnormal pollution spikes 
           using a rolling z-score detector (threshold: 4 sigma)?

        3. **Distribution shift** — Do sensor readings drift over time? 
           (Kolmogorov-Smirnov test per feature, p < 0.05 triggers retraining)

        4. **Environmental drivers** — How do temperature and humidity correlate with PM2.5 
           in Nairobi's climate context? (Pearson r, OLS regression)
        """)

    st.divider()

    # ── 1. Model accuracy over time
    st.subheader("1. Model Accuracy Over Time")
    if METRICS_PATH.exists():
        mdf = pd.read_csv(METRICS_PATH)
        if "timestamp" in mdf.columns:
            mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], errors="coerce")
        fig = go.Figure()
        if "test_mae" in mdf.columns:
            fig.add_trace(go.Scatter(
                x=mdf.get("timestamp", mdf.index), y=mdf["test_mae"],
                name="Test MAE", line=dict(color="#3b82f6", width=2)
            ))
        if "cv_mae_mean" in mdf.columns:
            fig.add_trace(go.Scatter(
                x=mdf.get("timestamp", mdf.index), y=mdf["cv_mae_mean"],
                name="CV MAE (mean)", line=dict(color="#8b5cf6", width=1.5, dash="dot")
            ))
        fig.update_layout(height=260, yaxis_title="MAE (ug/m3)", **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Lower MAE = better prediction accuracy. "
            f"Total retraining sessions recorded: {len(mdf)}. "
            f"Target: MAE < 5 ug/m3 for operational use."
        )
    else:
        st.info("No training history yet. Run the pipeline to generate metrics.")

    # ── 2. Data drift detection
    st.subheader("2. Data Drift Detection (Kolmogorov-Smirnov Test)")
    if DRIFT_PATH.exists():
        ddf = pd.read_csv(DRIFT_PATH)
        p_cols = [c for c in ddf.columns if c.endswith("_p")]
        if p_cols:
            fig2 = go.Figure()
            cmap = {
                "pm25_p": "#3b82f6", "pm10_p": "#f97316",
                "temperature_p": "#f43f5e", "humidity_p": "#06b6d4"
            }
            for pc in p_cols:
                fig2.add_trace(go.Scatter(
                    x=ddf.get("timestamp", ddf.index), y=ddf[pc],
                    name=pc.replace("_p", ""),
                    line=dict(color=cmap.get(pc, "#9ca3af"), width=1.5)
                ))
            fig2.add_hline(
                y=0.05, line=dict(color="#ef4444", dash="dash", width=1.5),
                annotation_text="Drift threshold (p = 0.05)",
                annotation_font=dict(color="#ef4444")
            )
            fig2.update_layout(height=250, yaxis_title="KS p-value", **CHART_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                "p-value below 0.05 indicates the current data distribution has shifted "
                "significantly from the reference window — the pipeline automatically retrains the model."
            )
    else:
        st.info("Drift reports will appear after at least two pipeline runs.")

    # ── 3. Temperature & Humidity impact
    st.subheader("3. Environmental Drivers of PM2.5")
    if not df.empty and "pm25" in df.columns:
        c1, c2 = st.columns(2)
        for feat, widget in [("temperature", c1), ("humidity", c2)]:
            if feat not in df.columns:
                continue
            sub  = df[["pm25", feat]].dropna().tail(2000)
            corr = float(sub["pm25"].corr(sub[feat]))
            strength  = "strong" if abs(corr)>0.5 else ("moderate" if abs(corr)>0.3 else "weak")
            direction = "positive" if corr > 0 else "negative"
            widget.metric(
                feat.capitalize(),
                f"r = {corr:.3f}",
                delta=f"{strength} {direction} correlation"
            )
            sample = sub.sample(min(600, len(sub)), random_state=42)
            try:
                fig_sc = px.scatter(
                    sample, x=feat, y="pm25", opacity=0.35,
                    trendline="ols",
                    color_discrete_sequence=["#3b82f6" if corr > 0 else "#ef4444"],
                    labels={feat: feat.capitalize(), "pm25": "PM2.5 (ug/m3)"},
                    height=250
                )
            except Exception:
                fig_sc = px.scatter(
                    sample, x=feat, y="pm25", opacity=0.35,
                    color_discrete_sequence=["#3b82f6"],
                    labels={feat: feat.capitalize(), "pm25": "PM2.5 (ug/m3)"},
                    height=250
                )
            fig_sc.update_layout(margin=dict(l=0,r=0,t=10,b=0),
                                 plot_bgcolor="#0a1628", paper_bgcolor="#0a1628",
                                 font_color="#94a3b8")
            widget.plotly_chart(fig_sc, use_container_width=True)

        st.caption(
            "Temperature: cooler air traps pollutants (temperature inversion), "
            "typically showing negative correlation. "
            "Humidity: hygroscopic growth of particles can increase measured PM2.5 "
            "at high humidity levels."
        )

    # ── 4. Feature importances
    st.subheader("4. Model Feature Importances")
    if FEAT_IMP_PATH.exists():
        fi = pd.read_csv(FEAT_IMP_PATH)
        fi_top = fi.head(12).copy()

        f1, f2 = st.columns([2, 1])
        with f1:
            fig_fi = px.bar(
                fi_top, x="importance", y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#1e3a5f", "#3b82f6"],
                height=340,
                labels={"importance": "Importance Score", "feature": "Feature"}
            )
            fig_fi.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="#0a1628", paper_bgcolor="#0a1628",
                font_color="#94a3b8", coloraxis_showscale=False
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        with f2:
            st.image(WIND_IMAGE, use_container_width=True)
            st.markdown(
                '<div class="img-caption">Wind speed and direction are key meteorological '
                'variables influencing PM2.5 dispersion across sensor locations</div>',
                unsafe_allow_html=True
            )
        st.caption(
            "Lag features (pm25_lag1-6) dominate — confirming strong temporal autocorrelation "
            "in PM2.5 time series. PM10 and rolling means provide additional predictive power."
        )
    else:
        st.info("Train the model to see feature importances.")

    # ── 5. PM2.5 distribution
    st.subheader("5. PM2.5 Concentration Distribution")
    if "pm25" in df.columns:
        sub = df["pm25"].dropna()
        fig_hist = px.histogram(
            sub, nbins=60,
            color_discrete_sequence=["#3b82f6"],
            labels={"value": "PM2.5 (ug/m3)", "count": "Frequency"},
            height=260
        )
        fig_hist.add_vline(x=12,   line=dict(color="#22c55e", dash="dash"),
                           annotation_text="Good", annotation_font=dict(color="#22c55e"))
        fig_hist.add_vline(x=35.4, line=dict(color="#eab308", dash="dash"),
                           annotation_text="Moderate", annotation_font=dict(color="#eab308"))
        fig_hist.add_vline(x=55.4, line=dict(color="#ef4444", dash="dash"),
                           annotation_text="Unhealthy", annotation_font=dict(color="#ef4444"))
        fig_hist.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="#0a1628", paper_bgcolor="#0a1628",
            font_color="#94a3b8", showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean PM2.5",    f"{sub.mean():.1f} ug/m3")
        c2.metric("Median PM2.5",  f"{sub.median():.1f} ug/m3")
        c3.metric("95th Percentile", f"{sub.quantile(0.95):.1f} ug/m3")
        c4.metric("Max PM2.5",     f"{sub.max():.1f} ug/m3")

        pct_good = 100 * (sub <= 12).sum() / len(sub)
        pct_mod  = 100 * ((sub > 12) & (sub <= 35.4)).sum() / len(sub)
        pct_unh  = 100 * (sub > 35.4).sum() / len(sub)
        st.caption(
            f"Time in Good range: {pct_good:.1f}% | "
            f"Moderate: {pct_mod:.1f}% | "
            f"Unhealthy or worse: {pct_unh:.1f}%"
        )

    # ── 6. Retraining history
    st.subheader("6. Automatic Retraining History")
    if RETRAIN_LOG_PATH.exists():
        try:
            logs = json.loads(RETRAIN_LOG_PATH.read_text())
            if logs:
                keep = [c for c in ["trained_at","model_type","n_train","n_test"] if c in logs[0]]
                st.dataframe(
                    pd.DataFrame(logs)[keep].tail(10),
                    use_container_width=True, hide_index=True
                )
        except Exception:
            st.info("Could not parse retraining log.")
    else:
        st.info("No retraining sessions logged yet.")


# ── Tab 4: Raw Data ───────────────────────────────────────────────────────────
def render_raw(df: pd.DataFrame):
    st.markdown('<div class="sec-hdr">Raw Data Explorer</div>', unsafe_allow_html=True)

    if df.empty:
        st.warning("No data loaded.")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    loc_opts = sorted(df["location"].unique().tolist()) if "location" in df.columns else []
    loc_filter = c1.multiselect("Filter by sensor location", options=loc_opts, default=[])
    n_rows = c2.slider("Rows to display", 50, 2000, 300, step=50)
    sort_col = c3.selectbox(
        "Sort by",
        options=[c for c in ["timestamp","pm25","pm10","temperature"] if c in df.columns],
        index=0
    )

    view = df.copy()
    if loc_filter:
        view = view[view["location"].isin(loc_filter)]
    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=False)

    default_cols = [c for c in
                    ["timestamp","location","pm25","pm10","pm1",
                     "temperature","humidity","pm25_spike"] if c in view.columns]
    selected_cols = st.multiselect("Columns to show", options=list(view.columns),
                                   default=default_cols)
    if selected_cols:
        st.dataframe(view[selected_cols].head(n_rows),
                     use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-hdr">Summary Statistics</div>', unsafe_allow_html=True)
    num_cols = [c for c in ["pm25","pm10","pm1","temperature","humidity"] if c in view.columns]
    if num_cols:
        st.dataframe(view[num_cols].describe().round(2), use_container_width=True)

    st.divider()
    st.download_button(
        "Download full dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"nairobi_air_quality_{datetime.utcnow().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.caption(f"Total records available: {len(df):,}  |  Showing: {min(n_rows, len(view)):,}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Header
    col_title, col_ts = st.columns([3, 1])
    with col_title:
        st.markdown(
            '<h1 style="color:#f1f5f9;font-size:22px;font-weight:700;margin-bottom:2px">'
            'Nairobi Air Quality Monitoring System</h1>'
            '<p style="color:#64748b;font-size:13px;margin:0">'
            'OpenAfrica sensorsAfrica &nbsp;|&nbsp; Self-updating ML pipeline &nbsp;|&nbsp; '
            'PM2.5 forecasting with Random Forest</p>',
            unsafe_allow_html=True
        )
    with col_ts:
        st.markdown(
            f'<div style="text-align:right;color:#475569;font-size:12px;padding-top:8px">'
            f'{datetime.utcnow().strftime("%d %b %Y %H:%M UTC")}</div>',
            unsafe_allow_html=True
        )

    df     = load_data()
    status = read_status()

    render_sidebar(status, df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Overview",
        "Sensor Map",
        "Research and Analysis",
        "Raw Data",
    ])

    with tab1:
        render_overview(df)
    with tab2:
        render_map(df)
    with tab3:
        render_research(df)
    with tab4:
        render_raw(df)


if __name__ == "__main__":
    main()