from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_file

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
LOGGER_SCRIPT = BACKEND_DIR / "data_logger.py"
LOGGER_DATA_DIR = PROJECT_DIR / "data" / "readings"
UPLOAD_DIR = PROJECT_DIR / "data" / "uploads"
PREDICTION_DIR = PROJECT_DIR / "data" / "predictions"
MODEL_PATH = PROJECT_DIR / "results" / "vibration_random_forest.pkl"
PREDICT_SCRIPT = BACKEND_DIR / "predict_vibration_fault.py"
LOGGER_URL = os.environ.get("LOGGER_URL", "http://127.0.0.1:8001")
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

for folder in (UPLOAD_DIR, PREDICTION_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
logger_start_lock = threading.Lock()
logger_process: subprocess.Popen | None = None


def logger_request(path: str, method: str = "GET", body: dict | None = None):
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if payload else {}
    req = urllib.request.Request(f"{LOGGER_URL}{path}", data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            return json.loads(response.read().decode("utf-8")), response.status
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read().decode("utf-8")), exc.code
        except (ValueError, UnicodeDecodeError):
            return {"error": "The logger returned an unreadable response."}, exc.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return {"error": "Logger is offline. Start data_logger.py and try again."}, 503


@app.get("/api/health")
def health():
    return jsonify({"ready": MODEL_PATH.is_file(), "logger_url": LOGGER_URL})


@app.post("/api/shutdown")
def shutdown_backend():
    """Stop the logger child and this local Python API after replying."""
    def stop_processes():
        global logger_process
        if logger_process is not None and logger_process.poll() is None:
            logger_process.terminate()
            try:
                logger_process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                logger_process.kill()
        os._exit(0)

    threading.Timer(0.6, stop_processes).start()
    return jsonify({"stopping": True, "message": "Stopping the Python backend and sensor logger."}), 202


@app.get("/api/logger/data")
def logger_data():
    body, status = logger_request("/api/data")
    return jsonify(body), status


@app.get("/api/logger/session/status")
def logger_session_status():
    body, status = logger_request("/api/session/status")
    return jsonify(body), status


@app.post("/api/logger/session/start")
def logger_session_start():
    body, status = logger_request("/api/session/start", "POST", request.get_json(silent=True) or {})
    return jsonify(body), status


@app.post("/api/logger/session/stop")
def logger_session_stop():
    body, status = logger_request("/api/session/stop", "POST", {})
    return jsonify(body), status


@app.post("/api/logger/activate")
def activate_logger():
    """Start the local sensor logger if it is not already responding."""
    global logger_process
    body, status = logger_request("/api/data")
    if status == 200:
        if body.get("serial_connected"):
            return jsonify({"active": True, "message": "Sensor backend is active and connected."})
        return jsonify({
            "active": True,
            "serial_connected": False,
            "message": body.get("serial_error") or "Logger is running, but the sensor is not connected. Check the configured COM port and close any other app using it.",
        })

    if not LOGGER_SCRIPT.is_file():
        return jsonify({"error": f"Logger script not found: {LOGGER_SCRIPT}"}), 404

    with logger_start_lock:
        body, status = logger_request("/api/data")
        if status == 200:
            if body.get("serial_connected"):
                return jsonify({"active": True, "message": "Sensor backend is active and connected."})
            return jsonify({
                "active": True,
                "serial_connected": False,
                "message": body.get("serial_error") or "Logger is running, but the sensor is not connected. Check the configured COM port and close any other app using it.",
            })
        try:
            log_path = BACKEND_DIR / "logger.log"
            log_handle = log_path.open("a", encoding="utf-8")
            flags = 0
            if os.name == "nt":
                flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            logger_process = subprocess.Popen(
                [sys.executable, str(LOGGER_SCRIPT)],
                cwd=BACKEND_DIR,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=flags,
                close_fds=True,
            )
            log_handle.close()
            return jsonify({"active": False, "starting": True, "message": "Starting the sensor backend. It may take a few seconds."}), 202
        except OSError as exc:
            return jsonify({"error": f"Could not start data_logger.py: {exc}"}), 500


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (stem or "vibration_readings")[:80]


def run_prediction(source_path: Path, output_path: Path) -> dict:
    if not PREDICT_SCRIPT.is_file():
        raise FileNotFoundError(f"Prediction program not found: {PREDICT_SCRIPT}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    check = pd.read_csv(source_path, nrows=2)
    if check.empty:
        raise ValueError("The CSV has no data rows.")
    result = subprocess.run(
        [sys.executable, str(PREDICT_SCRIPT), str(source_path), "--model", str(MODEL_PATH),
         "--output", str(output_path), "--probabilities"],
        cwd=PROJECT_DIR, capture_output=True, text=True, timeout=120, check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or "The model could not process this CSV.")
    table = pd.read_csv(output_path)
    table.insert(0, "row", range(1, len(table) + 1))
    records = json.loads(table.to_json(orient="records", date_format="iso"))
    return {"rows": records, "output_file": output_path.name, "message": result.stdout.strip()}


@app.post("/api/predict")
def predict():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose a CSV file first."}), 400
    if Path(uploaded.filename).suffix.lower() != ".csv":
        return jsonify({"error": "Please select a .csv file."}), 400
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = safe_stem(uploaded.filename)
    source_path = UPLOAD_DIR / f"{stem}_{stamp}.csv"
    output_path = PREDICTION_DIR / f"{stem}_{stamp}_predictions.csv"
    uploaded.save(source_path)

    try:
        return jsonify(run_prediction(source_path, output_path))
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        source_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return jsonify({"error": f"Could not read this CSV. Please check its format. Details: {exc}"}), 400
    except FileNotFoundError as exc:
        output_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 500
    except (ValueError, subprocess.TimeoutExpired) as exc:
        output_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc) if not isinstance(exc, subprocess.TimeoutExpired) else "Prediction took too long. Try a smaller CSV."}), 400
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.get("/api/recordings")
def recordings():
    files = []
    if LOGGER_DATA_DIR.is_dir():
        for path in sorted(LOGGER_DATA_DIR.glob("*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)[:100]:
            files.append({"name": path.name, "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes"), "bytes": path.stat().st_size})
    return jsonify(files)


DATA_FOLDERS = {
    "readings": PROJECT_DIR / "data" / "readings",
    "uploads": UPLOAD_DIR,
    "predictions": PREDICTION_DIR,
}


@app.get("/api/files")
def list_data_files():
    files = []
    for folder_name, folder_path in DATA_FOLDERS.items():
        folder_path.mkdir(parents=True, exist_ok=True)
        for path in folder_path.glob("*.csv"):
            stat = path.stat()
            files.append({
                "name": path.name,
                "folder": folder_name,
                "bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="minutes"),
            })
    files.sort(key=lambda item: item["modified"], reverse=True)
    return jsonify(files)


@app.get("/api/files/<folder_name>/<path:filename>")
def download_data_file(folder_name: str, filename: str):
    folder = DATA_FOLDERS.get(folder_name)
    if folder is None:
        return jsonify({"error": "Unknown data folder."}), 404
    target = (folder / Path(filename).name).resolve()
    if target.parent != folder.resolve() or not target.is_file() or target.suffix.lower() != ".csv":
        return jsonify({"error": "CSV file was not found."}), 404
    return send_file(target, as_attachment=True, download_name=target.name, mimetype="text/csv")


@app.post("/api/predict-saved")
def predict_saved_file():
    request_data = request.get_json(silent=True) or {}
    folder_name = str(request_data.get("folder", ""))
    filename = str(request_data.get("name", ""))
    folder = DATA_FOLDERS.get(folder_name)
    if folder is None:
        return jsonify({"error": "Choose a CSV from the data folders."}), 400
    source_path = (folder / Path(filename).name).resolve()
    if source_path.parent != folder.resolve() or not source_path.is_file() or source_path.suffix.lower() != ".csv":
        return jsonify({"error": "CSV file was not found. Refresh the file list and try again."}), 404

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = PREDICTION_DIR / f"{safe_stem(source_path.name)}_{stamp}_predictions.csv"
    try:
        return jsonify(run_prediction(source_path, output_path))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        output_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        output_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 500
    except subprocess.TimeoutExpired:
        output_path.unlink(missing_ok=True)
        return jsonify({"error": "Prediction took too long. Try a smaller CSV."}), 400
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.get("/api/predictions/<path:filename>")
def download_prediction(filename: str):
    target = (PREDICTION_DIR / Path(filename).name).resolve()
    if target.parent != PREDICTION_DIR.resolve() or not target.is_file():
        return jsonify({"error": "Prediction CSV was not found."}), 404
    return send_file(target, as_attachment=True, download_name=target.name, mimetype="text/csv")


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "This file is larger than the 30 MB upload limit."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
