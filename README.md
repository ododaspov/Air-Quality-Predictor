# Nairobi Real-Time Air Quality Monitoring System

A research-grade, self-updating air quality monitoring and forecasting pipeline built on live sensor data from the [OpenAfrica sensorsAfrica network](https://open.africa).


---

## Project Overview

This project addresses four research questions in real-time urban air quality science:

| Question | Approach |
|----------|----------|
| Can we predict PM2.5 one hour ahead? | Random Forest with lag + rolling features, TimeSeriesSplit CV |
| Can we detect abnormal pollution spikes? | Rolling z-score detector (4 sigma threshold) per sensor |
| Can we build a self-updating dashboard? | Streamlit + background scheduler polling every 5 minutes |
| Can we auto-retrain when new data arrives? | MD5 hash change detection + MAE degradation trigger |

---

## Applications

### 1. Research Dashboard (`dashboard.py`)
Full scientific monitoring dashboard with:
- Live KPI cards (PM2.5, PM10, PM1, Temperature, Humidity)
- 1-hour and 6-hour PM2.5 forecasts with confidence intervals
- Per-sensor pollution trend charts with WHO threshold lines
- Diurnal pattern analysis
- Interactive sensor map (Mapbox)
- Data drift detection (Kolmogorov-Smirnov test)
- Model accuracy tracking over time
- Feature importance visualisation
- Automatic retraining history

### 2. Respiratory Health Advisor (`health_advisor.py`)
Real-world application for users with respiratory conditions:
- 12 respiratory conditions with personalised PM2.5 thresholds
- 15 Nairobi neighbourhoods with nearest-sensor matching
- Activity-level adjusted effective exposure calculation
- Condition-specific precautionary measures
- Medication checklist before visits
- Emergency facility directory with phone numbers
- WHO guideline comparison chart

---

## Project Structure

```
Air_Qual_Project/
│
├── data_pipeline.py          # Download → pivot long/wide → clean → feature engineer
├── model_training.py         # Random Forest training, evaluation, prediction
├── auto_update.py            # Background scheduler: polls OpenAfrica every 5 min
├── dashboard.py              # Research Streamlit dashboard
├── health_advisor.py         # Respiratory Health Advisor Streamlit app
│
├── data_cleaning_eda.ipynb   # Interactive notebook: EDA, cleaning, baseline model
│
├── .env.example              # Environment variable template (copy to .env)
├── requirements.txt          # Python dependencies
├── setup.bat                 # Windows one-click setup
├── setup.sh                  # macOS/Linux one-click setup
└── README.md
```

---


## Dataset

**Source:** [OpenAfrica sensorsAfrica — Nairobi](https://open.africa)

**Format:** Long-format CSV with columns:
`sensor_id | sensor_type | location | lat | lon | timestamp | value_type | value`

**Sensor types:**
- `DHT22` → `humidity`, `temperature`
- `pms5003` → `P0` (PM1), `P1` (PM10), `P2` (PM2.5)

**Coverage:** 2 locations, 10 sensors, March 2026 onwards

---

## ML Model

| Component | Detail |
|-----------|--------|
| Algorithm | Random Forest Regressor (200 trees) |
| Target | PM2.5 value at next 10-minute interval |
| Features | temperature, humidity, PM10, PM2.5 lag 1–6, rolling mean/std, hour, day of week |
| Validation | TimeSeriesSplit (5 folds) — no data leakage |
| Retrain trigger | MAE > 10 µg/m³ on last 200 rows, OR every 12th data update |
| Test MAE | ~3.2 µg/m³ |
| Test R² | ~0.71 |

---

## WHO PM2.5 Thresholds

| Category | PM2.5 (µg/m³) |
|----------|--------------|
| Good | 0 – 12 |
| Moderate | 12.1 – 35.4 |
| Unhealthy for Sensitive Groups | 35.5 – 55.4 |
| Unhealthy | 55.5 – 150.4 |
| Very Unhealthy | 150.5 – 250.4 |
| Hazardous | > 250.4 |

---

## References

- WHO Global Air Quality Guidelines (2021)
- OpenAfrica sensorsAfrica initiative
- Zheng, T. et al. (2019). *Real-time PM2.5 forecasting using low-cost sensors.* Atmospheric Measurement Techniques
- Giannakis, E. et al. (2021). *Machine learning for urban air quality prediction in African cities*

---

## Disclaimer

This application provides air quality information to assist in decision-making. It is not a substitute for medical advice. In a medical emergency, call **999** or **112**.
