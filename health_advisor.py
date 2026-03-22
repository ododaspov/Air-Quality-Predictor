"""
health_advisor.py
─────────────────
Nairobi Respiratory Health Advisor
A real-world application that advises users with respiratory conditions
about air quality safety at Nairobi sensor locations.

Run: streamlit run health_advisor.py
"""

import os, json, time, pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nairobi Respiratory Health Advisor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
HIST_PATH   = Path(os.getenv("HIST_PATH",   "historical_data.csv"))
MODEL_PATH  = Path(os.getenv("MODEL_PATH",  "air_quality_model.pkl"))
STATUS_PATH = Path(os.getenv("STATUS_PATH", "update_status.json"))
DATA_URL    = os.getenv("DATA_URL", "")

# ── Unsplash images (no API key) ──────────────────────────────────────────────
IMG = {
    "hero":    "https://images.unsplash.com/photo-1569585723035-da1ae4b0ecab?w=1400&q=80",
    "lungs":   "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=600&q=80",
    "nairobi": "https://images.unsplash.com/photo-1611273426858-450d8e3c9fce?w=800&q=80",
    "mask":    "https://images.unsplash.com/photo-1583947215259-38e31be8751f?w=600&q=80",
    "doctor":  "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=600&q=80",
    "outdoor": "https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?w=600&q=80",
    "pills":   "https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?w=600&q=80",
    "clinic":  "https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?w=600&q=80",
}

# ── Respiratory conditions database ──────────────────────────────────────────
RESPIRATORY_CONDITIONS = {
    "Asthma": {
        "description": "Chronic inflammatory disease causing airway obstruction and bronchospasm.",
        "pm25_safe": 12.0,
        "pm25_caution": 25.0,
        "pm25_danger": 35.4,
        "triggers": ["PM2.5", "PM10", "ozone", "cold air", "smoke", "dust"],
        "risk_level": "high",
        "color": "#3b82f6",
        "medications_to_carry": ["Rescue inhaler (SABA)", "Preventer inhaler (ICS)"],
        "who_guideline": 15,
    },
    "Chronic Obstructive Pulmonary Disease (COPD)": {
        "description": "Progressive lung disease causing persistent airflow limitation.",
        "pm25_safe": 10.0,
        "pm25_caution": 20.0,
        "pm25_danger": 30.0,
        "triggers": ["PM2.5", "PM10", "smoke", "chemical fumes", "dust"],
        "risk_level": "very_high",
        "color": "#f97316",
        "medications_to_carry": ["Bronchodilator", "Corticosteroids", "Oxygen if prescribed"],
        "who_guideline": 10,
    },
    "Bronchiectasis": {
        "description": "Permanent widening of airways leading to mucus buildup and infection risk.",
        "pm25_safe": 10.0,
        "pm25_caution": 20.0,
        "pm25_danger": 30.0,
        "triggers": ["PM2.5", "bacteria", "smoke", "dust", "pollutants"],
        "risk_level": "very_high",
        "color": "#8b5cf6",
        "medications_to_carry": ["Antibiotics (if prescribed)", "Airway clearance device"],
        "who_guideline": 10,
    },
    "Pulmonary Fibrosis": {
        "description": "Scarring of lung tissue leading to progressive breathing difficulty.",
        "pm25_safe": 8.0,
        "pm25_caution": 15.0,
        "pm25_danger": 25.0,
        "triggers": ["PM2.5", "PM10", "smoke", "chemical exposure", "dust"],
        "risk_level": "very_high",
        "color": "#ef4444",
        "medications_to_carry": ["Anti-fibrotic medication", "Supplemental oxygen"],
        "who_guideline": 10,
    },
    "Allergic Rhinitis": {
        "description": "Inflammation of nasal passages caused by allergens and irritants.",
        "pm25_safe": 15.0,
        "pm25_caution": 35.4,
        "pm25_danger": 55.4,
        "triggers": ["PM2.5", "pollen", "dust", "smoke"],
        "risk_level": "moderate",
        "color": "#22c55e",
        "medications_to_carry": ["Antihistamines", "Nasal corticosteroid spray"],
        "who_guideline": 15,
    },
    "Pneumonia (recovering)": {
        "description": "Recovering from lung infection — temporarily heightened sensitivity.",
        "pm25_safe": 8.0,
        "pm25_caution": 15.0,
        "pm25_danger": 25.0,
        "triggers": ["PM2.5", "bacteria", "viruses", "cold air", "smoke"],
        "risk_level": "very_high",
        "color": "#f43f5e",
        "medications_to_carry": ["Prescribed antibiotics", "Bronchodilator if prescribed"],
        "who_guideline": 10,
    },
    "Cystic Fibrosis": {
        "description": "Genetic disorder causing thick mucus buildup in lungs and airways.",
        "pm25_safe": 8.0,
        "pm25_caution": 15.0,
        "pm25_danger": 25.0,
        "triggers": ["PM2.5", "bacteria", "smoke", "dust", "allergens"],
        "risk_level": "very_high",
        "color": "#06b6d4",
        "medications_to_carry": ["CFTR modulator", "Dornase alfa", "Airway clearance device"],
        "who_guideline": 10,
    },
    "Sleep Apnea": {
        "description": "Repeated airway obstruction during sleep, worsened by poor air quality.",
        "pm25_safe": 15.0,
        "pm25_caution": 35.4,
        "pm25_danger": 55.4,
        "triggers": ["PM2.5", "smoke", "allergens", "nasal congestion triggers"],
        "risk_level": "moderate",
        "color": "#a78bfa",
        "medications_to_carry": ["CPAP machine", "Prescribed medication"],
        "who_guideline": 15,
    },
    "Lung Cancer (treatment)": {
        "description": "Undergoing cancer treatment — severely reduced respiratory immunity.",
        "pm25_safe": 5.0,
        "pm25_caution": 12.0,
        "pm25_danger": 20.0,
        "triggers": ["All particulate matter", "smoke", "chemical fumes", "radon"],
        "risk_level": "critical",
        "color": "#dc2626",
        "medications_to_carry": ["Prescribed cancer medications", "Supplemental oxygen"],
        "who_guideline": 5,
    },
    "Tuberculosis (treatment)": {
        "description": "Active TB treatment — compromised lung function and immune response.",
        "pm25_safe": 10.0,
        "pm25_caution": 20.0,
        "pm25_danger": 30.0,
        "triggers": ["PM2.5", "dust", "smoke", "crowded environments"],
        "risk_level": "very_high",
        "color": "#b45309",
        "medications_to_carry": ["Anti-TB regimen medications", "Mask (protect others)"],
        "who_guideline": 10,
    },
    "Childhood Asthma": {
        "description": "Asthma in children — more vulnerable due to developing lungs.",
        "pm25_safe": 8.0,
        "pm25_caution": 15.0,
        "pm25_danger": 25.0,
        "triggers": ["PM2.5", "PM10", "smoke", "dust", "cold air", "exercise"],
        "risk_level": "very_high",
        "color": "#ec4899",
        "medications_to_carry": ["Paediatric inhaler", "Spacer device", "Written asthma plan"],
        "who_guideline": 10,
    },
    "Occupational Lung Disease": {
        "description": "Work-related lung damage (e.g. silicosis, asbestosis).",
        "pm25_safe": 10.0,
        "pm25_caution": 20.0,
        "pm25_danger": 30.0,
        "triggers": ["PM2.5", "PM10", "dust", "chemical fumes", "industrial pollutants"],
        "risk_level": "high",
        "color": "#64748b",
        "medications_to_carry": ["Bronchodilator", "Corticosteroids if prescribed"],
        "who_guideline": 10,
    },
}

# ── Nairobi neighbourhood → coordinates mapping ───────────────────────────────
NAIROBI_PLACES = {
    "Westlands":         (-1.2676, 36.8065),
    "Karen":             (-1.3189, 36.7144),
    "Kilimani":          (-1.2921, 36.7871),
    "Lavington":         (-1.2835, 36.7717),
    "Eastleigh":         (-1.2762, 36.8494),
    "Industrial Area":   (-1.3080, 36.8412),
    "Gigiri":            (-1.2295, 36.8042),
    "Parklands":         (-1.2625, 36.8148),
    "South B":           (-1.3128, 36.8356),
    "Lang'ata":          (-1.3468, 36.7545),
    "Ngong Road":        (-1.3012, 36.7760),
    "Thika Road":        (-1.2150, 36.8900),
    "Mombasa Road":      (-1.3340, 36.8540),
    "CBD (City Centre)": (-1.2864, 36.8172),
    "Ruiru":             (-1.1470, 36.9613),
}

# ── Risk assessment ───────────────────────────────────────────────────────────
def assess_risk(pm25: float, condition: str) -> dict:
    c = RESPIRATORY_CONDITIONS[condition]
    safe_t    = c["pm25_safe"]
    caution_t = c["pm25_caution"]
    danger_t  = c["pm25_danger"]

    if pm25 <= safe_t:
        level, label, color, icon = 0, "Safe", "#22c55e", "SAFE"
    elif pm25 <= caution_t:
        level, label, color, icon = 1, "Low Risk", "#84cc16", "LOW RISK"
    elif pm25 <= danger_t:
        level, label, color, icon = 2, "Moderate Risk", "#eab308", "CAUTION"
    elif pm25 <= danger_t * 1.5:
        level, label, color, icon = 3, "High Risk", "#f97316", "HIGH RISK"
    else:
        level, label, color, icon = 4, "Dangerous", "#ef4444", "DANGER"

    return {
        "level": level, "label": label, "color": color, "icon": icon,
        "pm25": pm25, "condition": condition,
        "safe_threshold": safe_t, "caution_threshold": caution_t,
        "danger_threshold": danger_t,
    }


def get_precautions(risk: dict, condition: str) -> list[str]:
    c     = RESPIRATORY_CONDITIONS[condition]
    level = risk["level"]
    pm25  = risk["pm25"]

    base = []
    if level == 0:
        base = [
            "Air quality is within safe limits for your condition today.",
            "Outdoor activities are generally appropriate — enjoy your visit.",
            "Continue regular medication schedule as prescribed.",
            "Monitor symptoms and carry your rescue medication as always.",
        ]
    elif level == 1:
        base = [
            "Air quality is slightly elevated but manageable with precautions.",
            "Limit strenuous outdoor exercise to short durations.",
            "Take your preventer medication as scheduled before going out.",
            "Carry your rescue inhaler or emergency medication at all times.",
            "Avoid areas with heavy traffic or visible smoke sources.",
        ]
    elif level == 2:
        base = [
            f"PM2.5 level ({pm25:.1f} ug/m3) exceeds your safe threshold ({c['pm25_safe']} ug/m3).",
            "Wear a well-fitted N95 or FFP2 mask outdoors.",
            "Limit outdoor exposure to under 30 minutes at a time.",
            "Avoid outdoor exercise — switch to indoor alternatives.",
            "Use air purifier indoors if available.",
            "Take pre-emptive bronchodilator dose if advised by your doctor.",
            "Monitor symptoms closely — leave the area if symptoms worsen.",
        ]
    elif level == 3:
        base = [
            f"PM2.5 level ({pm25:.1f} ug/m3) significantly exceeds your safe threshold.",
            "STRONGLY recommended to postpone non-essential outdoor activities.",
            "If going out is unavoidable, wear N95 mask and minimise time outdoors.",
            "Keep windows and doors closed — use air conditioning or purifier.",
            "Take all prescribed medications as directed.",
            "Contact your doctor or clinic if you develop any respiratory symptoms.",
            "Have emergency contact numbers ready.",
        ]
    else:
        base = [
            f"CRITICAL: PM2.5 level ({pm25:.1f} ug/m3) is at a dangerous level for your condition.",
            "Avoid all outdoor activity. Remain indoors with windows closed.",
            "Seek air-conditioned, filtered indoor environments immediately.",
            "Ensure all medications are taken as prescribed — do not skip doses.",
            "If experiencing breathing difficulty, contact emergency services.",
            "Alert a trusted person about your condition and location.",
            "Consider contacting your physician for temporary medication adjustment.",
        ]

    # Add condition-specific precautions
    if "Asthma" in condition:
        base.append("Identify and avoid local asthma triggers (vehicle exhaust, dust, smoke).")
        if level >= 2:
            base.append("Use your peak flow meter to monitor lung function during the visit.")
    if "COPD" in condition:
        base.append("Walk at a slower pace and rest frequently to avoid overexertion.")
        if level >= 2:
            base.append("If on supplemental oxygen, ensure adequate supply for the trip duration.")
    if "Fibrosis" in condition or "fibrosis" in condition:
        base.append("Avoid prolonged exposure to any dust or particulate sources.")
    if "Cancer" in condition or "TB" in condition or "tuberculosis" in condition.lower():
        base.append("Wear a medical-grade mask throughout and sanitise hands frequently.")
    if "Child" in condition:
        base.append("Children should not run or play outdoors in elevated pollution conditions.")

    return base


def get_medications_reminder(condition: str) -> list[str]:
    return RESPIRATORY_CONDITIONS[condition].get("medications_to_carry", [])


def get_nearby_facilities() -> list[dict]:
    """Return key Nairobi medical facilities relevant to respiratory emergencies."""
    return [
        {"name": "Kenyatta National Hospital",    "type": "National Referral",   "phone": "+254 20 272 6300", "area": "Upper Hill"},
        {"name": "Aga Khan University Hospital",  "type": "Private Hospital",    "phone": "+254 20 366 2000", "area": "Parklands"},
        {"name": "Nairobi Hospital",              "type": "Private Hospital",    "phone": "+254 20 284 5000", "area": "Upper Hill"},
        {"name": "MP Shah Hospital",              "type": "Private Hospital",    "phone": "+254 20 489 2000", "area": "Parklands"},
        {"name": "Gertrude's Children's Hospital","type": "Paediatric Hospital", "phone": "+254 20 720 4000", "area": "Muthaiga"},
        {"name": "Mater Hospital",                "type": "Private Hospital",    "phone": "+254 20 690 0000", "area": "South B"},
        {"name": "Emergency Services (Kenya)",    "type": "Emergency",           "phone": "999 / 112",       "area": "Nationwide"},
        {"name": "AMREF Flying Doctors",          "type": "Air Ambulance",       "phone": "+254 20 600 0090","area": "Wilson Airport"},
    ]


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_air_quality() -> pd.DataFrame:
    if not HIST_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(HIST_PATH, low_memory=False)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp")
    for col in ["pm25","pm10","pm1","temperature","humidity","lat","lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_location_aqi(df: pd.DataFrame, place_name: str) -> dict:
    """Get the latest AQI reading for a given place, finding the nearest sensor."""
    target_lat, target_lon = NAIROBI_PLACES[place_name]

    if df.empty or "lat" not in df.columns:
        return {"error": "No sensor data available"}

    # Get latest per-location readings
    latest = (
        df.dropna(subset=["pm25","lat","lon"])
          .sort_values("timestamp")
          .groupby("location")
          .last()
          .reset_index()
    )
    if latest.empty:
        return {"error": "No readings available"}

    # Find nearest sensor by Euclidean distance
    latest["dist"] = np.sqrt(
        (latest["lat"] - target_lat)**2 + (latest["lon"] - target_lon)**2
    )
    nearest = latest.loc[latest["dist"].idxmin()]
    dist_km = float(nearest["dist"]) * 111  # rough deg→km conversion

    # 24h trend for this location
    loc_id    = nearest["location"]
    loc_data  = df[df["location"] == loc_id].dropna(subset=["pm25"])
    cutoff    = loc_data["timestamp"].max() - pd.Timedelta(hours=24)
    last_24h  = loc_data[loc_data["timestamp"] >= cutoff]
    avg_24h   = float(last_24h["pm25"].mean()) if not last_24h.empty else None
    trend     = loc_data.tail(12)["pm25"].tolist()

    # 1h forecast
    forecast_pm25 = None
    if MODEL_PATH.exists():
        try:
            from data_pipeline import engineer_features
            from model_training import predict_next_hour
            feat_df = engineer_features(df[df["location"] == loc_id])
            if not feat_df.dropna(subset=["pm25"]).empty:
                row = feat_df.dropna(subset=["pm25"]).iloc[-1]
                res = predict_next_hour(row)
                forecast_pm25 = res["pm25_predicted"]
        except Exception:
            pass

    return {
        "place":        place_name,
        "sensor_id":    int(loc_id),
        "sensor_lat":   float(nearest["lat"]),
        "sensor_lon":   float(nearest["lon"]),
        "distance_km":  round(dist_km, 1),
        "pm25":         round(float(nearest["pm25"]), 1),
        "pm10":         round(float(nearest.get("pm10", 0) or 0), 1),
        "pm1":          round(float(nearest.get("pm1", 0) or 0), 1),
        "temperature":  round(float(nearest.get("temperature", 0) or 0), 1),
        "humidity":     round(float(nearest.get("humidity", 0) or 0), 1),
        "timestamp":    str(nearest.get("timestamp", ""))[:19],
        "avg_24h":      round(avg_24h, 1) if avg_24h else None,
        "trend":        trend,
        "forecast_1h":  round(forecast_pm25, 1) if forecast_pm25 else None,
    }


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stSidebar"] { display: none; }

/* Step cards */
.step-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 16px; padding: 28px 32px; margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.step-number {
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; border-radius: 50%;
    background: #0f172a; color: white;
    font-size: 14px; font-weight: 700; margin-bottom: 12px;
}
.step-title { font-size: 17px; font-weight: 700; color: #0f172a; margin-bottom: 4px; }
.step-sub   { font-size: 13px; color: #64748b; margin-bottom: 20px; }

/* Risk badge */
.risk-badge {
    display: inline-block; padding: 6px 20px; border-radius: 99px;
    font-size: 13px; font-weight: 700; letter-spacing: .06em;
}

/* AQI display */
.aqi-number {
    font-size: 64px; font-weight: 800; line-height: 1;
    letter-spacing: -2px;
}
.aqi-label { font-size: 14px; color: #64748b; margin-top: 4px; }

/* Precaution item */
.precaution {
    background: #f8fafc; border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    margin-bottom: 8px; font-size: 14px; color: #1e293b;
}
.precaution-danger { border-left-color: #ef4444; background: #fff5f5; }
.precaution-warn   { border-left-color: #f97316; background: #fff8f0; }
.precaution-safe   { border-left-color: #22c55e; background: #f0fdf4; }

/* Med card */
.med-card {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    font-size: 14px; color: #1e40af; font-weight: 500;
}

/* Facility row */
.facility-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid #f1f5f9;
    font-size: 13px;
}
.facility-name { font-weight: 600; color: #0f172a; }
.facility-type { color: #64748b; font-size: 12px; }
.facility-phone { color: #3b82f6; font-weight: 600; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 20px; padding: 40px 48px; margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.hero-title {
    font-size: 28px; font-weight: 800; color: white;
    line-height: 1.2; margin-bottom: 8px;
}
.hero-subtitle { font-size: 15px; color: #94a3b8; max-width: 600px; }
.hero-tag {
    display: inline-block; background: rgba(59,130,246,.2);
    color: #93c5fd; padding: 4px 12px; border-radius: 99px;
    font-size: 12px; font-weight: 600; letter-spacing: .05em;
    margin-bottom: 16px;
}

/* Metric pill */
.metric-pill {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 12px 16px; text-align: center;
}
.metric-pill-label { font-size: 11px; color: #94a3b8; text-transform: uppercase;
                      letter-spacing: .06em; margin-bottom: 4px; }
.metric-pill-value { font-size: 20px; font-weight: 700; color: #0f172a; }
.metric-pill-unit  { font-size: 11px; color: #94a3b8; margin-left: 2px; }

/* Condition card */
.cond-card {
    background: white; border: 2px solid #e2e8f0;
    border-radius: 12px; padding: 16px; cursor: pointer;
    transition: all .15s; text-align: center;
}
.cond-card:hover { border-color: #3b82f6; box-shadow: 0 4px 12px rgba(59,130,246,.15); }
.cond-card-name  { font-size: 13px; font-weight: 600; color: #0f172a; margin-bottom: 4px; }
.cond-card-risk  { font-size: 11px; color: #64748b; }
</style>
""", unsafe_allow_html=True)


# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">NAIROBI RESPIRATORY HEALTH ADVISOR</div>
    <div class="hero-title">Is it safe to visit?<br>Get personalised air quality advice.</div>
    <div class="hero-subtitle">
        Select your respiratory condition, choose a Nairobi neighbourhood,
        and receive real-time safety assessment powered by live sensor data
        and machine-learning forecasts.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Context images ────────────────────────────────────────────────────────────
ci1, ci2, ci3 = st.columns(3)
ci1.image(IMG["nairobi"], use_container_width=True, caption="Nairobi urban environment — multiple pollution sources")
ci2.image(IMG["lungs"],   use_container_width=True, caption="Respiratory conditions require personalised AQI thresholds")
ci3.image(IMG["mask"],    use_container_width=True, caption="Protective equipment recommendations based on real-time data")

st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_air_quality()
data_ok = not df.empty and "pm25" in df.columns

if not data_ok:
    st.error(
        "No air quality data found. Make sure historical_data.csv exists in the project folder. "
        "Run: python model_training.py"
    )
    st.stop()

# ── STEP 1: Select condition ──────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-number">1</div>
    <div class="step-title">Select Your Respiratory Condition</div>
    <div class="step-sub">Choose the condition that applies to you. Each condition has personalised
    PM2.5 thresholds calibrated to clinical guidelines.</div>
</div>
""", unsafe_allow_html=True)

condition = st.selectbox(
    "My respiratory condition:",
    options=list(RESPIRATORY_CONDITIONS.keys()),
    index=0,
    help="Select the respiratory condition you are managing."
)

cond_data = RESPIRATORY_CONDITIONS[condition]

# Condition info panel
ci_col1, ci_col2, ci_col3 = st.columns(3)
with ci_col1:
    st.markdown(f"""
    <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;padding:16px;height:100%">
        <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;
                    margin-bottom:6px">Condition</div>
        <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:8px">{condition}</div>
        <div style="font-size:13px;color:#475569">{cond_data['description']}</div>
    </div>
    """, unsafe_allow_html=True)

with ci_col2:
    risk_colors = {"moderate":"#eab308","high":"#f97316","very_high":"#ef4444","critical":"#dc2626"}
    rl = cond_data["risk_level"].replace("_", " ").upper()
    rc = risk_colors.get(cond_data["risk_level"], "#64748b")
    st.markdown(f"""
    <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;padding:16px">
        <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:10px">Your PM2.5 Thresholds</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px">
            <span style="font-size:13px;color:#475569">Safe below</span>
            <span style="font-size:13px;font-weight:700;color:#22c55e">
                {cond_data['pm25_safe']} ug/m3</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px">
            <span style="font-size:13px;color:#475569">Caution above</span>
            <span style="font-size:13px;font-weight:700;color:#eab308">
                {cond_data['pm25_caution']} ug/m3</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="font-size:13px;color:#475569">Danger above</span>
            <span style="font-size:13px;font-weight:700;color:#ef4444">
                {cond_data['pm25_danger']} ug/m3</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ci_col3:
    meds = cond_data.get("medications_to_carry", [])
    meds_html = "".join(
        f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;'
        f'padding:6px 12px;margin-bottom:6px;font-size:13px;color:#1e40af;font-weight:500">'
        f'{m}</div>'
        for m in meds
    )
    st.markdown(f"""
    <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;padding:16px">
        <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:10px">Always carry</div>
        {meds_html}
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── STEP 2: Choose destination ────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-number">2</div>
    <div class="step-title">Choose Your Destination in Nairobi</div>
    <div class="step-sub">Select the area you plan to visit. Air quality is matched to the
    nearest live sensor in the network.</div>
</div>
""", unsafe_allow_html=True)

place = st.selectbox(
    "I am visiting:",
    options=list(NAIROBI_PLACES.keys()),
    index=0,
)

# Also allow custom visit time
visit_col1, visit_col2 = st.columns(2)
with visit_col1:
    visit_duration = st.select_slider(
        "Planned outdoor duration",
        options=["< 15 min", "15-30 min", "30-60 min", "1-2 hours", "2-4 hours", "All day"],
        value="30-60 min",
    )
with visit_col2:
    activity_level = st.selectbox(
        "Activity level during visit",
        ["Resting / seated", "Light walking", "Moderate walking", "Brisk walking / light exercise",
         "Strenuous exercise"],
        index=1
    )

# Activity multiplier for effective exposure
ACTIVITY_MULTIPLIERS = {
    "Resting / seated": 1.0,
    "Light walking": 1.2,
    "Moderate walking": 1.5,
    "Brisk walking / light exercise": 1.8,
    "Strenuous exercise": 2.3,
}
act_mult = ACTIVITY_MULTIPLIERS[activity_level]

st.divider()

# ── STEP 3: Assessment ────────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-number">3</div>
    <div class="step-title">Your Personalised Safety Assessment</div>
    <div class="step-sub">Real-time air quality data from the nearest sensorsAfrica sensor,
    adjusted for your condition and activity level.</div>
</div>
""", unsafe_allow_html=True)

aqi_data = get_location_aqi(df, place)

if "error" in aqi_data:
    st.error(f"Could not retrieve sensor data: {aqi_data['error']}")
    st.stop()

raw_pm25      = aqi_data["pm25"]
effective_pm25 = round(raw_pm25 * act_mult, 1)  # adjusted for breathing rate
risk          = assess_risk(effective_pm25, condition)
precautions   = get_precautions(risk, condition)

# ── Main assessment display
left, right = st.columns([1, 2])

with left:
    # AQI Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=raw_pm25,
        number={"suffix": " ug/m3", "font": {"size": 28, "color": "#0f172a"}},
        gauge={
            "axis": {"range": [0, 150], "tickcolor": "#94a3b8",
                     "tickfont": {"color": "#94a3b8", "size": 10}},
            "bar":  {"color": risk["color"], "thickness": 0.25},
            "bgcolor": "#f8fafc",
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, cond_data["pm25_safe"]],    "color": "#dcfce7"},
                {"range": [cond_data["pm25_safe"], cond_data["pm25_caution"]], "color": "#fef9c3"},
                {"range": [cond_data["pm25_caution"], cond_data["pm25_danger"]], "color": "#fed7aa"},
                {"range": [cond_data["pm25_danger"], 150], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": risk["color"], "width": 3},
                "thickness": 0.8,
                "value": raw_pm25,
            },
        },
        title={"text": f"PM2.5 at {place}", "font": {"size": 14, "color": "#475569"}},
    ))
    fig_gauge.update_layout(
        height=280, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="white", font_color="#0f172a"
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Risk badge
    st.markdown(
        f'<div style="text-align:center;margin-top:8px">'
        f'<div style="font-size:28px;font-weight:800;color:{risk["color"]}">'
        f'{risk["label"]}</div>'
        f'<div style="font-size:13px;color:#64748b;margin-top:4px">'
        f'for {condition}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Effective exposure notice
    if act_mult > 1.0:
        st.markdown(
            f'<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:10px;'
            f'padding:10px 14px;margin-top:12px;font-size:12px;color:#92400e">'
            f'<strong>Effective exposure:</strong> {effective_pm25} ug/m3 '
            f'(raw {raw_pm25} x {act_mult} activity factor for {activity_level})</div>',
            unsafe_allow_html=True
        )

with right:
    # Sensor data pills
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, val, unit in [
        (m1, "PM2.5",    raw_pm25,            "ug/m3"),
        (m2, "PM10",     aqi_data["pm10"],     "ug/m3"),
        (m3, "PM1",      aqi_data["pm1"],      "ug/m3"),
        (m4, "Temp",     aqi_data["temperature"],"degC"),
        (m5, "Humidity", aqi_data["humidity"], "%"),
    ]:
        col.markdown(
            f'<div class="metric-pill">'
            f'<div class="metric-pill-label">{label}</div>'
            f'<div class="metric-pill-value">{val}'
            f'<span class="metric-pill-unit">{unit}</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div style="font-size:11px;color:#94a3b8;margin-top:6px">'
        f'Nearest sensor: {aqi_data["distance_km"]} km from {place} &nbsp;|&nbsp; '
        f'Reading: {aqi_data["timestamp"]} UTC &nbsp;|&nbsp; '
        f'24h average: {aqi_data.get("avg_24h","--")} ug/m3</div>',
        unsafe_allow_html=True
    )

    # 1h forecast callout
    if aqi_data.get("forecast_1h"):
        fc = aqi_data["forecast_1h"]
        fc_risk = assess_risk(fc, condition)
        delta   = fc - raw_pm25
        arrow   = "improving" if delta < -1 else ("worsening" if delta > 1 else "stable")
        fc_col  = "#22c55e" if delta < -1 else ("#ef4444" if delta > 1 else "#eab308")
        st.markdown(
            f'<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:12px;'
            f'padding:14px 18px;margin-top:12px">'
            f'<div style="font-size:12px;color:#0369a1;font-weight:600;letter-spacing:.04em;'
            f'text-transform:uppercase;margin-bottom:6px">1-Hour ML Forecast</div>'
            f'<div style="display:flex;align-items:center;gap:16px">'
            f'<div style="font-size:28px;font-weight:800;color:{fc_col}">{fc} ug/m3</div>'
            f'<div style="font-size:13px;color:#0369a1">Conditions forecast to be '
            f'<strong>{arrow}</strong> in 1 hour<br>'
            f'Forecast risk: <strong style="color:{fc_risk["color"]}">{fc_risk["label"]}</strong>'
            f'</div></div></div>',
            unsafe_allow_html=True
        )

    # 12-point trend sparkline
    if aqi_data.get("trend") and len(aqi_data["trend"]) > 2:
        trend_vals = aqi_data["trend"]
        fig_spark = go.Figure()
        fig_spark.add_trace(go.Scatter(
            y=trend_vals, mode="lines+markers",
            line=dict(color=risk["color"], width=2),
            marker=dict(size=4, color=risk["color"]),
            fill="tozeroy", fillcolor=f"rgba({int(risk['color'][1:3],16)},{int(risk['color'][3:5],16)},{int(risk['color'][5:7],16)},0.15)",
        ))
        fig_spark.add_hline(
            y=cond_data["pm25_safe"],
            line=dict(color="#22c55e", dash="dash", width=1),
            annotation_text="Your safe limit"
        )
        fig_spark.update_layout(
            height=120, margin=dict(l=0, r=0, t=4, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(visible=False), yaxis=dict(showgrid=False, visible=True, tickfont=dict(size=9)),
            showlegend=False,
        )
        st.markdown('<div style="font-size:12px;color:#64748b;margin-top:12px;margin-bottom:4px">'
                    'Recent trend (last 12 readings)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_spark, use_container_width=True)

st.divider()

# ── STEP 4: Precautions ───────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-number">4</div>
    <div class="step-title">Precautionary Measures</div>
    <div class="step-sub">Personalised actions based on your condition, the current
    air quality, and your planned activity.</div>
</div>
""", unsafe_allow_html=True)

prec_col1, prec_col2 = st.columns([3, 2])

with prec_col1:
    prec_css = {0: "precaution-safe", 1: "precaution-safe",
                2: "precaution-warn", 3: "precaution-warn", 4: "precaution-danger"}
    css_class = prec_css.get(risk["level"], "precaution")

    for i, p in enumerate(precautions):
        num_color = "#22c55e" if risk["level"] <= 1 else ("#f97316" if risk["level"] <= 3 else "#ef4444")
        st.markdown(
            f'<div class="precaution {css_class}">'
            f'<span style="font-weight:700;color:{num_color};margin-right:8px">{i+1}.</span>'
            f'{p}</div>',
            unsafe_allow_html=True
        )

with prec_col2:
    img_key = "outdoor" if risk["level"] <= 1 else ("mask" if risk["level"] == 2 else "doctor")
    st.image(IMG[img_key], use_container_width=True)
    captions = {
        "outdoor": "Safe conditions — outdoor activity is appropriate with standard precautions.",
        "mask":    "Elevated pollution — wear N95 mask and limit outdoor exposure time.",
        "doctor":  "High-risk conditions — consult your physician before outdoor activities.",
    }
    st.markdown(f'<div style="font-size:11px;color:#94a3b8;text-align:center;margin-top:4px">'
                f'{captions[img_key]}</div>', unsafe_allow_html=True)

    # Duration-based warning
    duration_risk = {
        "< 15 min": "Low exposure duration — manageable with precautions.",
        "15-30 min": "Moderate duration — monitor symptoms throughout.",
        "30-60 min": "Extended duration — rest frequently and use mask.",
        "1-2 hours": "Long exposure — consider breaking into shorter segments.",
        "2-4 hours": "Very long exposure — strongly consider postponing if risk > Moderate.",
        "All day":   "Full-day exposure — only advisable if risk level is Safe or Low.",
    }
    dur_warn = duration_risk.get(visit_duration, "")
    if dur_warn:
        bg = "#f0fdf4" if risk["level"] <= 1 else ("#fff7ed" if risk["level"] <= 2 else "#fff5f5")
        bc = "#22c55e" if risk["level"] <= 1 else ("#f97316" if risk["level"] <= 2 else "#ef4444")
        st.markdown(
            f'<div style="background:{bg};border-left:3px solid {bc};border-radius:0 8px 8px 0;'
            f'padding:10px 14px;margin-top:12px;font-size:13px;color:#0f172a">'
            f'<strong>Duration note:</strong> {dur_warn}</div>',
            unsafe_allow_html=True
        )

st.divider()

# ── Medication reminder ───────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px">'
    'Medication Checklist for This Visit</div>',
    unsafe_allow_html=True
)
med_col1, med_col2 = st.columns([2, 1])
with med_col1:
    for med in get_medications_reminder(condition):
        st.checkbox(f"{med}", value=False, key=f"med_{med}")
    st.markdown(
        '<div style="font-size:12px;color:#94a3b8;margin-top:8px">'
        'Tick each item before leaving. Never visit without your emergency medication.</div>',
        unsafe_allow_html=True
    )
with med_col2:
    st.image(IMG["pills"], use_container_width=True,
             caption="Always carry prescribed medication during outdoor visits")

st.divider()

# ── Sensor map for destination ────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px">'
    f'Air Quality Map — {place}</div>',
    unsafe_allow_html=True
)

latest_all = (
    df.dropna(subset=["pm25","lat","lon"])
      .sort_values("timestamp")
      .groupby("location")
      .last()
      .reset_index()
)
target_lat, target_lon = NAIROBI_PLACES[place]

latest_all["AQI Cat"]   = latest_all["pm25"].apply(lambda v: assess_risk(v, condition)["label"])
latest_all["Dot Size"]  = latest_all["pm25"].clip(10, 200)
latest_all["Dot Color"] = latest_all["pm25"].apply(
    lambda v: assess_risk(v, condition)["color"]
)

map_fig = go.Figure()

# All sensor bubbles
for _, row in latest_all.iterrows():
    r = assess_risk(float(row["pm25"]), condition)
    map_fig.add_trace(go.Scattermapbox(
        lat=[row["lat"]], lon=[row["lon"]],
        mode="markers",
        marker=dict(size=20, color=r["color"], opacity=0.85),
        text=f"Sensor {int(row['location'])}<br>PM2.5: {row['pm25']:.1f} ug/m3<br>Risk: {r['label']}",
        hoverinfo="text",
        name=f"Sensor {int(row['location'])}",
        showlegend=True,
    ))

# Target location pin
map_fig.add_trace(go.Scattermapbox(
    lat=[target_lat], lon=[target_lon],
    mode="markers+text",
    marker=dict(size=16, color="#0f172a", symbol="star"),
    text=[place], textposition="top right",
    textfont=dict(size=13, color="#0f172a"),
    hoverinfo="text",
    name=f"Your destination: {place}",
    showlegend=True,
))

map_fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=dict(lat=target_lat, lon=target_lon),
        zoom=11,
    ),
    height=400,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="white",
    legend=dict(bgcolor="white", bordercolor="#e2e8f0", borderwidth=1,
                font=dict(size=12)),
)
st.plotly_chart(map_fig, use_container_width=True)

st.divider()

# ── Emergency facilities ──────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:4px">'
    'Nearby Respiratory Care Facilities</div>'
    '<div style="font-size:13px;color:#64748b;margin-bottom:16px">'
    'Save these numbers before your visit.</div>',
    unsafe_allow_html=True
)

fac_col1, fac_col2 = st.columns([3, 1])
with fac_col1:
    facilities = get_nearby_facilities()
    for fac in facilities:
        emergency = fac["type"] == "Emergency"
        bg = "#fff5f5" if emergency else "white"
        bc = "#fca5a5" if emergency else "#e2e8f0"
        st.markdown(
            f'<div style="background:{bg};border:1px solid {bc};border-radius:10px;'
            f'padding:12px 16px;margin-bottom:8px;display:flex;'
            f'justify-content:space-between;align-items:center">'
            f'<div>'
            f'<div style="font-size:14px;font-weight:600;color:#0f172a">{fac["name"]}</div>'
            f'<div style="font-size:12px;color:#64748b">{fac["type"]} &nbsp;|&nbsp; {fac["area"]}</div>'
            f'</div>'
            f'<div style="font-size:14px;font-weight:700;color:#3b82f6">{fac["phone"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

with fac_col2:
    st.image(IMG["clinic"], use_container_width=True,
             caption="Nairobi has several specialist respiratory care centres")
    st.markdown(
        '<div style="background:#fff5f5;border:1px solid #fca5a5;border-radius:10px;'
        'padding:12px;margin-top:8px;font-size:13px;color:#991b1b;font-weight:600;'
        'text-align:center">EMERGENCY<br>999 &nbsp;|&nbsp; 112</div>',
        unsafe_allow_html=True
    )

st.divider()

# ── WHO guideline comparison ──────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px">'
    'How Does Today Compare to WHO Guidelines?</div>',
    unsafe_allow_html=True
)

who_val  = cond_data["who_guideline"]
who_comp = go.Figure()
categories = ["WHO Annual\nGuideline", "WHO 24h\nGuideline", "Your Safe\nLimit",
               "Your Caution\nThreshold", "Today's\nPM2.5"]
values     = [5, 15, cond_data["pm25_safe"], cond_data["pm25_caution"], raw_pm25]
bar_colors = ["#22c55e", "#84cc16", "#3b82f6", "#eab308",
              "#22c55e" if raw_pm25 <= cond_data["pm25_safe"]
              else ("#eab308" if raw_pm25 <= cond_data["pm25_caution"] else "#ef4444")]

who_comp.add_trace(go.Bar(
    x=categories, y=values, marker_color=bar_colors,
    text=[f"{v} ug/m3" for v in values], textposition="outside",
    textfont=dict(size=12, color="#0f172a"),
))
who_comp.update_layout(
    height=280, margin=dict(l=0, r=0, t=20, b=0),
    paper_bgcolor="white", plot_bgcolor="white",
    yaxis=dict(title="PM2.5 (ug/m3)", gridcolor="#f1f5f9", tickcolor="#94a3b8"),
    xaxis=dict(tickcolor="#94a3b8"),
    font_color="#475569", showlegend=False,
)
st.plotly_chart(who_comp, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="background:#f8fafc;border-radius:12px;padding:20px 24px;margin-top:24px;'
    f'border:1px solid #e2e8f0">'
    f'<div style="font-size:13px;font-weight:600;color:#0f172a;margin-bottom:6px">'
    f'Medical Disclaimer</div>'
    f'<div style="font-size:12px;color:#64748b;line-height:1.6">'
    f'This application provides air quality information to assist respiratory patients in '
    f'making informed decisions. It is not a substitute for medical advice. '
    f'Always follow guidance from your healthcare provider. '
    f'In a medical emergency, call 999 or 112 immediately.<br><br>'
    f'Data source: OpenAfrica sensorsAfrica network &nbsp;|&nbsp; '
    f'Sensor data updated every 5 minutes &nbsp;|&nbsp; '
    f'Last loaded: {datetime.utcnow().strftime("%d %b %Y %H:%M UTC")}'
    f'</div></div>',
    unsafe_allow_html=True
)