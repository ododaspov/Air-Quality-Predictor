"""
model_training.py
─────────────────
Research-grade ML pipeline for PM2.5 forecasting — Nairobi.
All configuration loaded from .env
"""
 
import json, logging, os, pickle, warnings
from datetime import datetime
from pathlib import Path
 
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")
 
# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()
 
MODEL_PATH            = Path(os.getenv("MODEL_PATH",       "air_quality_model.pkl"))
METRICS_PATH          = Path(os.getenv("METRICS_PATH",     "model_metrics.csv"))
FEAT_IMP_PATH         = Path(os.getenv("FEAT_IMP_PATH",    "feature_importances.csv"))
RETRAIN_LOG_PATH      = Path(os.getenv("RETRAIN_LOG_PATH", "retrain_log.json"))
RETRAIN_MAE_THRESHOLD = float(os.getenv("RETRAIN_MAE_THRESHOLD", 10.0))
MODEL_TYPE            = os.getenv("MODEL_TYPE", "rf")
 
FEATURE_COLS = [
    "temperature", "humidity", "pm10",
    "pm25_lag1", "pm25_lag2", "pm25_lag3",
    "pm25_lag4", "pm25_lag5", "pm25_lag6",
    "pm10_lag1", "pm25_roll3", "pm25_roll6", "pm25_std6",
    "hour", "day_of_week",
]
TARGET_COL = "pm25_target"
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("model_training")
 
 
def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [c for c in FEATURE_COLS if c in df.columns]
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target '{TARGET_COL}' missing. Run engineer_features first.")
    sub = df[available + [TARGET_COL]].dropna()
    log.info(f"  Training set: {len(sub):,} rows  {len(available)} features")
    return sub[available], sub[TARGET_COL]
 
 
def evaluate(y_true, y_pred, label="test") -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    log.info(f"  [{label}] MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 4)}
 
 
def append_metrics(metrics: dict) -> None:
    row = {"timestamp": datetime.utcnow().isoformat(), **metrics}
    df  = pd.DataFrame([row])
    if METRICS_PATH.exists():
        df.to_csv(METRICS_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(METRICS_PATH, index=False)
 
 
def save_feature_importances(model, feature_names: list) -> pd.DataFrame:
    est = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    if not hasattr(est, "feature_importances_"):
        return pd.DataFrame()
    imp = pd.DataFrame({"feature": feature_names, "importance": est.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(FEAT_IMP_PATH, index=False)
    log.info(f"  Top features: {imp.head(3)['feature'].tolist()}")
    return imp
 
 
def train(df: pd.DataFrame, model_type: str = MODEL_TYPE) -> dict:
    X, y = prepare_xy(df)
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
 
    estimator = (
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                  min_samples_leaf=5, subsample=0.8, random_state=42)
        if model_type == "gb"
        else RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3,
                                   n_jobs=-1, random_state=42)
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
 
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes, cv_r2s = [], []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        pipeline.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        m = evaluate(y_train.iloc[va_idx], pipeline.predict(X_train.iloc[va_idx]), f"fold-{fold+1}")
        cv_maes.append(m["mae"]); cv_r2s.append(m["r2"])
 
    cv_metrics = {
        "cv_mae_mean": round(float(np.mean(cv_maes)), 3),
        "cv_mae_std":  round(float(np.std(cv_maes)),  3),
        "cv_r2_mean":  round(float(np.mean(cv_r2s)),  4),
        "model_type":  model_type,
    }
 
    pipeline.fit(X_train, y_train)
    test_metrics = evaluate(y_test, pipeline.predict(X_test), "test")
    fi = save_feature_importances(pipeline, X.columns.tolist())
 
    meta = {
        "trained_at": datetime.utcnow().isoformat(), "model_type": model_type,
        "n_train": len(X_train), "n_test": len(X_test),
        "features": X.columns.tolist(), "cv_metrics": cv_metrics, "test_metrics": test_metrics,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": pipeline, "meta": meta, "feature_cols": X.columns.tolist()}, f)
    log.info(f"  Model saved → {MODEL_PATH}")
 
    append_metrics({**cv_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}})
    _update_retrain_log(meta)
 
    return {"cv_metrics": cv_metrics, "test_metrics": test_metrics,
            "feature_importances": fi.to_dict("records") if not fi.empty else [],
            "model_path": str(MODEL_PATH), "n_samples": len(X)}
 
 
def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run train() first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
 
 
def _who_category(pm25: float) -> tuple[str, str]:
    if pm25 <= 12:    return "Good", "green"
    if pm25 <= 35.4:  return "Moderate", "yellow"
    if pm25 <= 55.4:  return "Unhealthy for Sensitive Groups", "orange"
    if pm25 <= 150.4: return "Unhealthy", "red"
    if pm25 <= 250.4: return "Very Unhealthy", "purple"
    return "Hazardous", "maroon"
 
 
def predict_next_hour(latest_row) -> dict:
    artifact  = load_model()
    pipeline  = artifact["pipeline"]
    feat_cols = artifact["feature_cols"]
    row = pd.Series(latest_row) if isinstance(latest_row, dict) else latest_row
    # Only keep the feature columns we need, coerce to numeric, fill NaNs with column mean
    row_features = pd.to_numeric(row.reindex(feat_cols), errors="coerce")
    numeric_mean = row_features.mean()
    X = pd.DataFrame([row_features.fillna(numeric_mean)])
    pred = max(0.0, float(pipeline.predict(X)[0]))
 
    ci_lo = ci_hi = pred
    est = pipeline.named_steps.get("model")
    if hasattr(est, "estimators_"):
        scaler = pipeline.named_steps["scaler"]
        tree_preds = np.array([t.predict(scaler.transform(X))[0] for t in est.estimators_])
        ci_lo = float(np.percentile(tree_preds, 5))
        ci_hi = float(np.percentile(tree_preds, 95))
 
    cat, color = _who_category(pred)
    return {
        "pm25_predicted": round(pred, 2), "ci_low": round(max(0, ci_lo), 2),
        "ci_high": round(ci_hi, 2), "who_category": cat, "alert_color": color,
        "is_dangerous": pred > 55.4, "predicted_at": datetime.utcnow().isoformat(),
    }
 
 
def predict_batch(df: pd.DataFrame) -> pd.Series:
    artifact  = load_model()
    pipeline  = artifact["pipeline"]
    feat_cols = artifact["feature_cols"]
    available = [c for c in feat_cols if c in df.columns]
    X = df[available].fillna(df[available].mean())
    return pd.Series(np.maximum(0, pipeline.predict(X)), index=X.index)
 
 
def should_retrain(df: pd.DataFrame) -> tuple[bool, str]:
    if not MODEL_PATH.exists():
        return True, "No model exists yet."
    try:
        available = [c for c in FEATURE_COLS if c in df.columns]
        if TARGET_COL not in df.columns or len(available) < 5:
            return False, "Insufficient features to evaluate."
        sub = df[available + [TARGET_COL]].dropna().tail(200)
        if len(sub) < 20:
            return False, "Too few recent rows."
        mae = float(mean_absolute_error(sub[TARGET_COL], predict_batch(sub)))
        if mae > RETRAIN_MAE_THRESHOLD:
            return True, f"Recent MAE={mae:.2f} exceeds threshold ({RETRAIN_MAE_THRESHOLD})"
        return False, f"Recent MAE={mae:.2f} within threshold."
    except Exception as e:
        return True, f"Evaluation error: {e}"
 
 
def _update_retrain_log(meta: dict) -> None:
    log_data = []
    if RETRAIN_LOG_PATH.exists():
        try:
            log_data = json.loads(RETRAIN_LOG_PATH.read_text())
        except Exception:
            pass
    log_data.append(meta)
    RETRAIN_LOG_PATH.write_text(json.dumps(log_data[-50:], indent=2))
 
 
def temp_humidity_impact(df: pd.DataFrame) -> dict:
    if "pm25" not in df.columns:
        return {}
    report = {}
    for feat in ["temperature", "humidity"]:
        if feat not in df.columns:
            continue
        sub  = df[["pm25", feat]].dropna()
        corr = float(sub["pm25"].corr(sub[feat]))
        report[feat] = {
            "pearson_r": round(corr, 3),
            "direction": "positive" if corr > 0 else "negative",
            "strength":  "strong" if abs(corr) > 0.5 else ("moderate" if abs(corr) > 0.3 else "weak"),
        }
    return report
 
 
if __name__ == "__main__":
    from data_pipeline import run_pipeline
    print("Running pipeline…")
    feat_df, stats = run_pipeline()
    print(f"Pipeline stats: {stats}")
    print("\nTraining model…")
    result = train(feat_df)
    print(f"\nCV  MAE: {result['cv_metrics']['cv_mae_mean']} ± {result['cv_metrics']['cv_mae_std']}")
    print(f"Test MAE: {result['test_metrics']['mae']}")
    print(f"Test R² : {result['test_metrics']['r2']}")
    print("\nTemp/Humidity impact:")
    for feat, info in temp_humidity_impact(feat_df).items():
        print(f"  {feat}: r={info['pearson_r']}  ({info['strength']} {info['direction']})")
        