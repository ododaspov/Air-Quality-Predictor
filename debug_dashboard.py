"""
debug_dashboard.py
──────────────────
Run this INSTEAD of dashboard.py to diagnose why nothing shows.
    streamlit run debug_dashboard.py
"""
import os, json, pickle
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Debug", layout="wide")
st.title("🔧 Dashboard Debug Tool")

HIST_PATH    = Path(os.getenv("HIST_PATH",    "historical_data.csv"))
MODEL_PATH   = Path(os.getenv("MODEL_PATH",   "air_quality_model.pkl"))
STATUS_PATH  = Path(os.getenv("STATUS_PATH",  "update_status.json"))

# ── 1. File existence ─────────────────────────────────────────
st.subheader("1. File Check")
files = {
    "historical_data.csv": HIST_PATH,
    "air_quality_model.pkl": MODEL_PATH,
    "update_status.json": STATUS_PATH,
    ".env": Path(".env"),
    "data_pipeline.py": Path("data_pipeline.py"),
    "model_training.py": Path("model_training.py"),
}
for name, path in files.items():
    exists = path.exists()
    size   = f"{path.stat().st_size:,} bytes" if exists else "—"
    st.write(f"{'✅' if exists else '❌'} `{name}` — {size}")

# ── 2. DATA_URL from .env ─────────────────────────────────────
st.subheader("2. Environment Variables")
url = os.getenv("DATA_URL","NOT SET")
st.write(f"**DATA_URL:** `{url[:80]}...`" if len(url)>80 else f"**DATA_URL:** `{url}`")

# ── 3. historical_data.csv preview ───────────────────────────
st.subheader("3. historical_data.csv")
if HIST_PATH.exists():
    try:
        df = pd.read_csv(HIST_PATH, low_memory=False)
        st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        st.write("**Columns:**", list(df.columns))
        st.write("**dtypes:**")
        st.write(df.dtypes.astype(str).to_dict())
        st.write("**Missing values:**")
        st.write(df.isnull().sum().to_dict())
        st.write("**Head:**")
        st.dataframe(df.head(5))

        # Try timestamp parse
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            bad_ts = df["timestamp"].isna().sum()
            st.write(f"Timestamp parse — bad rows: {bad_ts}")
            st.write(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

        # Check pm25 values
        if "pm25" in df.columns:
            df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
            st.write(f"PM2.5 — valid: {df['pm25'].notna().sum()}  "
                     f"mean: {df['pm25'].mean():.2f}  "
                     f"min: {df['pm25'].min():.2f}  "
                     f"max: {df['pm25'].max():.2f}")
        else:
            st.error("❌ 'pm25' column NOT FOUND in CSV!")

    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
else:
    st.error("❌ historical_data.csv does not exist!")
    st.info("Run: python model_training.py   to generate it.")

# ── 4. Model pickle ───────────────────────────────────────────
st.subheader("4. Model File")
if MODEL_PATH.exists():
    try:
        with open(MODEL_PATH,"rb") as f:
            art = pickle.load(f)
        meta = art.get("meta",{})
        st.success("Model loaded successfully")
        st.write("**Features:**", art.get("feature_cols",[]))
        st.write("**Trained at:**", meta.get("trained_at","?"))
        st.write("**Test metrics:**", meta.get("test_metrics",{}))
    except Exception as e:
        st.error(f"Model load failed: {e}")
else:
    st.warning("❌ No model file found.")

# ── 5. Pipeline status ────────────────────────────────────────
st.subheader("5. Pipeline Status")
if STATUS_PATH.exists():
    st.json(json.loads(STATUS_PATH.read_text()))
else:
    st.warning("No status file found.")

# ── 6. Import test ────────────────────────────────────────────
st.subheader("6. Import Test")
imports_ok = True
for mod in ["pandas","numpy","plotly","sklearn","scipy","dotenv","requests"]:
    try:
        __import__(mod)
        st.write(f"✅ {mod}")
    except ImportError as e:
        st.error(f"❌ {mod} — {e}")
        imports_ok = False

# ── 7. engineer_features test ─────────────────────────────────
st.subheader("7. engineer_features() Test")
if HIST_PATH.exists():
    try:
        from data_pipeline import engineer_features
        df2 = pd.read_csv(HIST_PATH, low_memory=False)
        df2["timestamp"] = pd.to_datetime(df2["timestamp"], utc=True, errors="coerce")
        df2["pm25"] = pd.to_numeric(df2["pm25"], errors="coerce")
        feat = engineer_features(df2)
        st.success(f"engineer_features() OK → shape {feat.shape}")
        st.write("**Feature columns added:**",
                 [c for c in feat.columns if "lag" in c or "roll" in c or "target" in c])
        latest = feat.dropna(subset=["pm25"]).iloc[-1]
        st.write("**Latest row (for prediction):**")
        st.write(latest.to_dict())
    except Exception as e:
        st.error(f"engineer_features() failed: {e}")
        import traceback
        st.code(traceback.format_exc())

# ── 8. Prediction test ────────────────────────────────────────
st.subheader("8. predict_next_hour() Test")
if MODEL_PATH.exists() and HIST_PATH.exists():
    try:
        from data_pipeline import engineer_features
        from model_training import predict_next_hour
        df3 = pd.read_csv(HIST_PATH, low_memory=False)
        df3["timestamp"] = pd.to_datetime(df3["timestamp"], utc=True, errors="coerce")
        df3["pm25"] = pd.to_numeric(df3["pm25"], errors="coerce")
        feat3 = engineer_features(df3)
        row = feat3.dropna(subset=["pm25"]).iloc[-1]
        result = predict_next_hour(row)
        st.success(f"Prediction OK: {result}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")
st.info("Copy the output above and share it — this tells us exactly what's failing in dashboard.py")
