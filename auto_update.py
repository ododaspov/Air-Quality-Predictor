"""
auto_update.py
──────────────
Background scheduler: polls OpenAfrica, runs pipeline, triggers retraining.
Windows-safe: ASCII-only log messages, UTF-8 file handler.
"""

import hashlib, json, logging, os, threading, time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

from data_pipeline import run_pipeline, DATA_URL
from model_training import train, should_retrain

POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", 300))
FORCE_RETRAIN_EVERY_N = int(os.getenv("FORCE_RETRAIN_EVERY_N", 12))
STATUS_PATH           = Path(os.getenv("STATUS_PATH",  "update_status.json"))
REQUEST_TIMEOUT       = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 60))
HASH_PATH             = Path(".last_data_hash")

# Windows-safe logging: UTF-8 file + ASCII-only stream
_file_handler   = logging.FileHandler("auto_update.log", encoding="utf-8")
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

log = logging.getLogger("auto_update")
log.setLevel(logging.INFO)
log.addHandler(_file_handler)
log.addHandler(_stream_handler)
log.propagate = False

_update_count = 0
_stop_event   = threading.Event()
_scheduler_thread = None


def _remote_hash(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return hashlib.md5(resp.content).hexdigest()
    except Exception as e:
        log.warning(f"Hash check failed: {e}")
        return None


def _saved_hash() -> str:
    return HASH_PATH.read_text().strip() if HASH_PATH.exists() else ""


def _save_hash(h: str) -> None:
    HASH_PATH.write_text(h)


def _write_status(status: dict) -> None:
    status["updated_at"] = datetime.utcnow().isoformat()
    STATUS_PATH.write_text(json.dumps(status, indent=2))


def read_status() -> dict:
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text())
        except Exception:
            pass
    return {"state": "initialising", "updated_at": None}


def run_update_cycle(force: bool = False) -> dict:
    global _update_count
    log.info("Update cycle started")
    _write_status({"state": "checking", "message": "Checking OpenAfrica for new data..."})

    new_hash     = _remote_hash(DATA_URL)
    data_changed = force or (new_hash is not None and new_hash != _saved_hash())

    if not data_changed:
        msg = "No new data detected."
        log.info(msg)
        _write_status({"state": "idle", "message": msg})
        return {"state": "idle", "message": msg}

    log.info("New data detected - running pipeline...")
    _write_status({"state": "downloading", "message": "New data detected - downloading and cleaning..."})

    try:
        feat_df, pipeline_stats = run_pipeline()
    except Exception as e:
        msg = f"Pipeline failed: {e}"
        log.error(msg)
        _write_status({"state": "error", "message": msg})
        return {"state": "error", "message": msg}

    _save_hash(new_hash or _saved_hash())
    _update_count += 1

    retrain_needed, reason = should_retrain(feat_df)
    force_periodic = (_update_count % FORCE_RETRAIN_EVERY_N == 0)

    if retrain_needed or force_periodic:
        reason_str = reason if retrain_needed else f"Periodic retrain (every {FORCE_RETRAIN_EVERY_N} updates)"
        log.info(f"Retraining: {reason_str}")
        _write_status({"state": "training", "message": f"Retraining model... ({reason_str})"})
        try:
            result = train(feat_df)
            status = {
                "state": "ready", "message": "Model retrained successfully.",
                "retrained": True, "retrain_reason": reason_str,
                "test_mae": result["test_metrics"]["mae"],
                "test_r2":  result["test_metrics"]["r2"],
                "cv_mae":   result["cv_metrics"]["cv_mae_mean"],
                "n_samples": result["n_samples"],
                "pipeline": pipeline_stats,
                "top_features": [f["feature"] for f in result["feature_importances"][:5]],
            }
        except Exception as e:
            msg = f"Training failed: {e}"
            log.error(msg)
            _write_status({"state": "error", "message": msg})
            return {"state": "error", "message": msg}
    else:
        status = {
            "state": "ready", "message": "Data updated. Model retained (performance OK).",
            "retrained": False, "skip_reason": reason, "pipeline": pipeline_stats,
        }

    _write_status(status)
    log.info(f"Cycle complete: {status['message']}")
    return status


def _scheduler_loop(interval: int) -> None:
    log.info(f"Scheduler started - polling every {interval}s")
    # On first start, only retrain if model doesn't exist yet; skip costly download
    try:
        from model_training import MODEL_PATH
        if not MODEL_PATH.exists():
            run_update_cycle(force=True)
        else:
            _write_status({"state": "ready", "message": "Dashboard loaded. Next sync in 5 min."})
    except Exception as e:
        log.error(f"Initial cycle error: {e}")
        _write_status({"state": "ready", "message": "Dashboard loaded. Scheduler running."})

    while not _stop_event.is_set():
        _stop_event.wait(timeout=interval)
        if _stop_event.is_set():
            break
        try:
            run_update_cycle()
        except Exception as e:
            log.error(f"Cycle error: {e}")
            _write_status({"state": "error", "message": str(e)})


def start_scheduler(interval: int = POLL_INTERVAL_SECONDS) -> threading.Thread:
    global _scheduler_thread, _stop_event
    if _scheduler_thread and _scheduler_thread.is_alive():
        return _scheduler_thread
    _stop_event.clear()
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop, args=(interval,), daemon=True, name="AQScheduler"
    )
    _scheduler_thread.start()
    log.info("Background scheduler launched.")
    return _scheduler_thread


def stop_scheduler() -> None:
    _stop_event.set()


if __name__ == "__main__":
    import sys
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else POLL_INTERVAL_SECONDS
    print(f"Starting auto-update loop (interval={interval}s) - Ctrl+C to stop")
    try:
        _scheduler_loop(interval)
    except KeyboardInterrupt:
        print("\nStopped.")
