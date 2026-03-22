"""
data_pipeline.py
────────────────
OpenAfrica Air-Quality Pipeline — Nairobi
Handles LONG-FORMAT sensorsAfrica data:
  sensor_id | sensor_type | location | lat | lon | timestamp | value_type | value

Two sensor types per location:
  DHT22 / similar → humidity, temperature
  pms5003 / similar → P0 (PM1), P1 (PM10), P2 (PM2.5)

Pipeline: download → parse → pivot → merge → clean → drift detect → export
All configuration loaded from .env
"""

import os, hashlib, logging, warnings
from datetime import datetime
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

DATA_URL        = os.getenv("DATA_URL")
HIST_PATH       = Path(os.getenv("HIST_PATH",        "historical_data.csv"))
AUDIT_PATH      = Path(os.getenv("AUDIT_LOG_PATH",   "pipeline_audit.log"))
DRIFT_PATH      = Path(os.getenv("DRIFT_PATH",       "drift_report.csv"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 30))

if not DATA_URL:
    raise EnvironmentError("DATA_URL not set. Check your .env file.")

# sensorsAfrica value_type names → standard column names
PM_TYPES  = ["P0", "P1", "P2"]
ENV_TYPES = ["temperature", "humidity"]
PM_RENAME = {"P0": "pm1", "P1": "pm10", "P2": "pm25"}

BOUNDS = {
    "pm25":        (0.0, 500.0),
    "pm10":        (0.0, 1000.0),
    "pm1":         (0.0, 400.0),
    "temperature": (-10.0, 60.0),
    "humidity":    (0.0, 100.0),
}
REQUIRED_FEATURES = ["pm25", "pm10", "temperature", "humidity"]

logging.basicConfig(
    handlers=[logging.FileHandler(AUDIT_PATH), logging.StreamHandler()],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")


# ── Download ──────────────────────────────────────────────────────────────────

def download_raw(url: str = DATA_URL) -> pd.DataFrame:
    """Download CSV from OpenAfrica and return raw long-format DataFrame."""
    log.info(f"Fetching: {url}")
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    text = resp.text
    log.info(f"Downloaded {len(text):,} bytes")

    delim = ";" if text.count(";") > text.count(",") else ","
    log.info(f"Detected delimiter: {repr(delim)}")

    # Use python engine + on_bad_lines to handle malformed rows gracefully
    df = pd.read_csv(
        StringIO(text),
        sep=delim,
        on_bad_lines="warn",
        engine="python",
    )
    log.info(f"Raw shape: {df.shape}  columns: {list(df.columns)}")
    return df


# ── Long → Wide transform ─────────────────────────────────────────────────────

def pivot_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert sensorsAfrica long-format data to wide format.

    Two sensor types post to the same location at slightly different times.
    Strategy:
      1. Parse timestamp + numeric value
      2. Pivot PM stream and ENV stream separately
      3. Resample both to 10-minute bins
      4. Merge on location + timestamp (outer join)
      5. Forward-fill short gaps within each location
    """
    # ── Parse types
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Standardise column names across dataset variants
    df = df.rename(columns={
        "measured_at": "timestamp",
        "datetime":    "timestamp",
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["value"]     = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"])

    log.info(f"  Unique value_types: {df['value_type'].unique().tolist()}")
    log.info(f"  Unique locations:   {df['location'].nunique()}")

    # ── Split into PM and ENV streams
    df_pm  = df[df["value_type"].isin(PM_TYPES)].copy()
    df_env = df[df["value_type"].isin(ENV_TYPES)].copy()
    log.info(f"  PM rows: {len(df_pm):,}   ENV rows: {len(df_env):,}")

    # ── Pivot each stream
    df_pm_wide = df_pm.pivot_table(
        index=["location", "lat", "lon", "timestamp"],
        columns="value_type",
        values="value",
        aggfunc="mean",
    ).reset_index()
    df_pm_wide.columns.name = None
    df_pm_wide = df_pm_wide.rename(columns=PM_RENAME)

    df_env_wide = df_env.pivot_table(
        index=["location", "timestamp"],
        columns="value_type",
        values="value",
        aggfunc="mean",
    ).reset_index()
    df_env_wide.columns.name = None

    # ── Resample to 10-minute bins
    def _resample(frame, value_cols, key_cols):
        frame = frame.copy()
        frame["timestamp"] = frame["timestamp"].dt.floor("10min")
        return (
            frame.groupby(key_cols + ["timestamp"])[value_cols]
                 .mean()
                 .reset_index()
        )

    pm_10  = _resample(df_pm_wide,  ["pm1", "pm10", "pm25"],      ["location", "lat", "lon"])
    env_10 = _resample(df_env_wide, ["temperature", "humidity"],   ["location"])

    log.info(f"  PM resampled: {pm_10.shape}   ENV resampled: {env_10.shape}")

    # ── Merge on location + timestamp
    merged = pd.merge(pm_10, env_10, on=["location", "timestamp"], how="outer")
    merged = merged.sort_values(["location", "timestamp"]).reset_index(drop=True)

    # ── Forward-fill within each location (max 3 steps = 30 min gap)
    fill_cols = ["pm25", "pm10", "pm1", "temperature", "humidity"]
    for col in fill_cols:
        if col in merged.columns:
            merged[col] = (
                merged.groupby("location")[col]
                      .transform(lambda s: s.ffill(limit=3).bfill(limit=3))
            )

    log.info(f"  Wide shape: {merged.shape}")
    missing_pct = merged[fill_cols].isnull().mean().round(3) * 100
    log.info(f"  Missing %%: {missing_pct.to_dict()}")

    return merged


# ── Physical range cleaning ───────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply physical bounds, remove duplicates, flag spikes."""
    df = df.copy()

    # Numeric coercion
    for col in BOUNDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physical range clipping
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            before  = df[col].notna().sum()
            df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi))
            removed = before - df[col].notna().sum()
            if removed > 0:
                log.info(f"  [{col}] clipped {removed} out-of-range values")

    # Drop rows with no PM2.5 (our target — useless without it)
    before = len(df)
    df = df.dropna(subset=["pm25"]).reset_index(drop=True)
    log.info(f"  Dropped {before - len(df)} rows with no PM2.5")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["location", "timestamp"], keep="last")
    if len(df) < before:
        log.info(f"  Removed {before - len(df)} duplicate rows")

    # Spike detection — rolling z-score > 4σ on PM2.5 per location
    if "pm25" in df.columns and len(df) > 20:
        def _spike_flag(s):
            rm  = s.rolling(12, min_periods=3).mean()
            rs  = s.rolling(12, min_periods=3).std().replace(0, np.nan)
            return ((s - rm).abs() / rs) > 4
        df["pm25_spike"] = (
            df.groupby("location")["pm25"]
              .transform(_spike_flag)
              .fillna(False)
        )
    else:
        df["pm25_spike"] = False

    log.info(f"  Clean shape: {df.shape}  spikes: {df['pm25_spike'].sum()}")
    return df


# ── Data drift detection ──────────────────────────────────────────────────────

def detect_drift(current: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """KS-test comparing current batch vs reference window."""
    results = {}
    for feat in ["pm25", "pm10", "temperature", "humidity"]:
        if feat not in current.columns or feat not in reference.columns:
            continue
        cur = current[feat].dropna().values
        ref = reference[feat].dropna().values
        if len(cur) < 10 or len(ref) < 10:
            continue
        stat, p_value = ks_2samp(ref, cur)
        drifted = p_value < 0.05
        results[feat] = {
            "ks_stat": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": drifted,
        }
        if drifted:
            log.warning(f"  DATA DRIFT [{feat}]: KS={stat:.3f}  p={p_value:.4f}")

    if results:
        row = {"timestamp": datetime.utcnow().isoformat()}
        for k, v in results.items():
            row[f"{k}_ks"]    = v["ks_stat"]
            row[f"{k}_p"]     = v["p_value"]
            row[f"{k}_drift"] = int(v["drift_detected"])
        drift_df = pd.DataFrame([row])
        if DRIFT_PATH.exists():
            drift_df.to_csv(DRIFT_PATH, mode="a", header=False, index=False)
        else:
            drift_df.to_csv(DRIFT_PATH, index=False)
    return results


# ── Incremental historical update ─────────────────────────────────────────────

def update_historical(new_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Merge cleaned wide data into historical_data.csv."""
    if HIST_PATH.exists():
        hist = pd.read_csv(HIST_PATH, low_memory=False)
        if "timestamp" in hist.columns:
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
        log.info(f"Loaded {len(hist):,} existing records from {HIST_PATH}")
    else:
        hist = pd.DataFrame()
        log.info("No historical file — creating fresh.")

    if hist.empty:
        combined, n_added = new_df, len(new_df)
    else:
        detect_drift(new_df, hist.tail(500))
        # Deduplicate on location + timestamp
        if "timestamp" in hist.columns and "timestamp" in new_df.columns:
            hist["_key"] = hist["location"].astype(str) + "_" + hist["timestamp"].astype(str)
            new_df = new_df.copy()
            new_df["_key"] = new_df["location"].astype(str) + "_" + new_df["timestamp"].astype(str)
            new_only = new_df[~new_df["_key"].isin(set(hist["_key"]))].drop(columns=["_key"])
            hist = hist.drop(columns=["_key"])
        else:
            new_only = new_df
        n_added  = len(new_only)
        combined = pd.concat([hist, new_only], ignore_index=True)

    combined.to_csv(HIST_PATH, index=False)
    log.info(f"  Saved {len(combined):,} records → {HIST_PATH}  (+{n_added} new)")
    return combined, n_added


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features, rolling statistics, and temporal features per location.
    Adds pm25_target (next PM2.5 reading) as the ML label.
    """
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)

    if "pm25" not in df.columns:
        return df

    # Compute features within each location group to avoid cross-location leakage
    def _add_features(g):
        g = g.sort_values("timestamp").copy()
        for lag in range(1, 7):
            g[f"pm25_lag{lag}"] = g["pm25"].shift(lag)
        if "pm10" in g.columns:
            g["pm10_lag1"] = g["pm10"].shift(1)
        g["pm25_roll3"] = g["pm25"].rolling(3,  min_periods=1).mean()
        g["pm25_roll6"] = g["pm25"].rolling(6,  min_periods=1).mean()
        g["pm25_std6"]  = g["pm25"].rolling(6,  min_periods=1).std()
        g["pm25_target"] = g["pm25"].shift(-1)   # next reading = ~10 min ahead
        return g

    df = df.groupby("location", group_keys=False).apply(_add_features)

    # Temporal features
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["hour"]        = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["month"]       = ts.dt.month

    return df.reset_index(drop=True)


# ── Public entry-point ────────────────────────────────────────────────────────

def run_pipeline(url: str = DATA_URL) -> tuple[pd.DataFrame, dict]:
    """
    Full pipeline:
      download → pivot long→wide → clean → update historical → feature engineering
    Returns (feature_engineered_df, pipeline_stats).
    """
    log.info("═" * 60)
    log.info("Pipeline run started")

    raw_df   = download_raw(url)
    wide_df  = pivot_long_to_wide(raw_df)
    clean_df = clean(wide_df)
    full_df, n_new = update_historical(clean_df)

    stats = {
        "run_time":   datetime.utcnow().isoformat(),
        "total_rows": len(full_df),
        "new_rows":   n_new,
        "locations":  int(full_df["location"].nunique()) if "location" in full_df.columns else None,
        "pm25_mean":  round(float(full_df["pm25"].mean()), 2) if "pm25" in full_df.columns else None,
        "pm25_std":   round(float(full_df["pm25"].std()),  2) if "pm25" in full_df.columns else None,
        "spikes":     int(full_df.get("pm25_spike", pd.Series([False])).sum()),
    }

    feat_df = engineer_features(full_df)
    log.info(f"Pipeline complete: {stats}")
    return feat_df, stats


if __name__ == "__main__":
    df, stats = run_pipeline()
    print("\n── Pipeline Stats ──")
    for k, v in stats.items():
        print(f"  {k}: {v}")
