"""
CNC Vibration Data Logger
--------------------------
Reads per-axis feature data from the ESP32 over USB serial, manages
labeled collection sessions, and writes one CSV file per session.

Run:
    python data_logger.py

Dashboard:
    http://127.0.0.1:8001

Install:
    pip install flask pyserial

== Serial format expected from firmware ==

Each data cycle produces 5 lines, separated by a '---...' line:

    ---------------------------------------
    Cycle #N | Fs=1657.0Hz | I2C errors(total)=N
    [X] RMS=0.1234g Peak=0.5678g Crest=4.610 Kurt=2.100 Skew=-0.300 DomFreq=123.5Hz Entropy=0.870
    [Y] RMS=...
    [Z] RMS=...

== CSV output schema (27 columns) ==

    session_id, label, pc_timestamp, esp_timestamp_ms, actual_fs,
    rms_x, peak_x, kurtosis_x, skewness_x, dom_freq_x, spectral_entropy_x,
    rms_y, peak_y, kurtosis_y, skewness_y, dom_freq_y, spectral_entropy_y,
    rms_z, peak_z, kurtosis_z, skewness_z, dom_freq_z, spectral_entropy_z,
    i2c_errors,
    cf_x, cf_y, cf_z

    - Columns 6-23 (rms_x ... spectral_entropy_z) are the 18 model inputs.
    - actual_fs: per-window measured Fs from the firmware header.
    - cf_x/y/z are Crest Factor, display-only — never feed these to the model.
    - i2c_errors is the PER-WINDOW delta of the firmware's cumulative counter.

== Stage 3 inference notes (train-serving skew prevention) ==

    The firmware computes:
      spectral_entropy /= log(nBins - 1)  where nBins = SAMPLES / 2 = 128
                       => log(127), NOT log(128)

    dom_freq = maxBin * (actualFs / SAMPLES)  using the MEASURED Fs per window.

    Any deviation in either formula causes silent train-serving skew.
"""

import csv
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime

import serial          # pip install pyserial
from flask import Flask, jsonify, render_template_string, request


# ============================================================
# Configuration — edit these before running
# ============================================================

SERIAL_PORT = "COM7"    # Windows: COM3, COM4, etc. Linux: /dev/ttyUSB0
SERIAL_BAUD = 115200

WEB_HOST = "0.0.0.0"
WEB_PORT = 8001

# Must match firmware #define SAMPLES (used to estimate esp_timestamp_ms)
FIRMWARE_SAMPLES = 256

VALID_LABELS = frozenset({"Normal", "Imbalance", "Misalignment", "Looseness"})

PROJECT_DIR = os.environ.get(
    "VIBRATION_ML_PROJECT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "readings")

# Ordered column list — order here IS the CSV column order
CSV_COLUMNS = [
    "session_id",
    "label",
    "pc_timestamp",
    "esp_timestamp_ms",
    "actual_fs",
    # 18 model input features (6 per axis × 3 axes)
    "rms_x",   "peak_x",   "kurtosis_x",   "skewness_x",   "dom_freq_x",   "spectral_entropy_x",
    "rms_y",   "peak_y",   "kurtosis_y",   "skewness_y",   "dom_freq_y",   "spectral_entropy_y",
    "rms_z",   "peak_z",   "kurtosis_z",   "skewness_z",   "dom_freq_z",   "spectral_entropy_z",
    # data quality
    "i2c_errors",
    # display-only — NOT model inputs
    "cf_x", "cf_y", "cf_z",
]

# ============================================================
# Firmware serial output parsers
# ============================================================

# "Cycle #42 | Fs=1657.0Hz | I2C errors(total)=0"
_RE_HEADER = re.compile(
    r"Cycle\s+#(\d+)\s*\|\s*Fs=([\d.]+)Hz\s*\|\s*I2C errors\(total\)=(\d+)"
)

# "[X] RMS=0.1234g Peak=0.5678g Crest=4.610 Kurt=2.100 Skew=-0.300 DomFreq=123.5Hz Entropy=0.870"
_RE_AXIS = re.compile(
    r"\[([XYZ])\]\s+RMS=([\d.]+)g\s+Peak=([\d.]+)g\s+Crest=([\d.]+)"
    r"\s+Kurt=(-?[\d.]+)\s+Skew=(-?[\d.]+)\s+DomFreq=([\d.]+)Hz\s+Entropy=([\d.]+)"
)


def _parse_cycle(lines):
    """Parse the lines accumulated between two '---' separators.

    Returns a flat dict with cycle/actual_fs/total_i2c_errors and all per-axis
    feature keys, or None if the required lines are missing or malformed.
    """
    header = None
    axes = {}

    for line in lines:
        m = _RE_HEADER.search(line)
        if m:
            header = m
            continue
        m = _RE_AXIS.search(line)
        if m:
            ax = m.group(1).lower()
            axes[ax] = {
                "rms":              float(m.group(2)),
                "peak":             float(m.group(3)),
                "cf":               float(m.group(4)),
                "kurtosis":         float(m.group(5)),
                "skewness":         float(m.group(6)),
                "dom_freq":         float(m.group(7)),
                "spectral_entropy": float(m.group(8)),
            }

    if header is None or not all(ax in axes for ax in ("x", "y", "z")):
        return None

    result = {
        "cycle":            int(header.group(1)),
        "actual_fs":        float(header.group(2)),
        "total_i2c_errors": int(header.group(3)),
    }
    for ax in ("x", "y", "z"):
        for feat in ("rms", "peak", "cf", "kurtosis", "skewness",
                     "dom_freq", "spectral_entropy"):
            result[f"{feat}_{ax}"] = axes[ax][feat]

    return result


# ============================================================
# Shared state
# ============================================================

data_lock = threading.Lock()
csv_lock  = threading.Lock()

# Latest decoded window — feature keys match CSV_COLUMNS where applicable
latest_data = {
    "serial_connected": False,
    "serial_error": None,
    "last_update":      None,
    "cycle":            None,
    "actual_fs":        None,
    "i2c_errors":       None,
}
for _col in CSV_COLUMNS[4:]:      # rms_x … cf_z
    latest_data[_col] = None

session_state = {
    "recording":   False,
    "session_id":  None,
    "label":       None,
    "row_count":   0,
    "error_count": 0,
    "csv_path":    None,
    "_writer":     None,       # csv.DictWriter, open only while recording
    "_file":       None,       # underlying file object
}


# ============================================================
# CSV helpers
# ============================================================

def _open_session_csv(session_id, label):
    os.makedirs(DATA_DIR, exist_ok=True)
    fname = f"{label}_{session_id}.csv"
    path = os.path.join(DATA_DIR, fname)
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    f.flush()
    return f, writer, path


def write_row(parsed, window_errors):
    """Append one CSV row. parsed is the dict from _parse_cycle."""
    cycle     = parsed["cycle"]
    actual_fs = parsed["actual_fs"]
    esp_ts    = round(cycle * FIRMWARE_SAMPLES / actual_fs * 1000) if actual_fs > 0 else 0

    with csv_lock:
        if not session_state["recording"]:
            return
        try:
            row = OrderedDict()
            row["session_id"]       = session_state["session_id"]
            row["label"]            = session_state["label"]
            row["pc_timestamp"]     = datetime.now().isoformat()
            row["esp_timestamp_ms"] = esp_ts
            row["actual_fs"]        = parsed["actual_fs"]

            for ax in ("x", "y", "z"):
                row[f"rms_{ax}"]              = parsed[f"rms_{ax}"]
                row[f"peak_{ax}"]             = parsed[f"peak_{ax}"]
                row[f"kurtosis_{ax}"]         = parsed[f"kurtosis_{ax}"]
                row[f"skewness_{ax}"]         = parsed[f"skewness_{ax}"]
                row[f"dom_freq_{ax}"]         = parsed[f"dom_freq_{ax}"]
                row[f"spectral_entropy_{ax}"] = parsed[f"spectral_entropy_{ax}"]

            row["i2c_errors"] = window_errors

            for ax in ("x", "y", "z"):
                row[f"cf_{ax}"] = parsed[f"cf_{ax}"]

            session_state["_writer"].writerow(row)
            session_state["_file"].flush()
            session_state["row_count"] += 1

        except Exception as exc:
            session_state["error_count"] += 1
            print(f"[CSV ERROR] {exc}", file=sys.stderr)


# ============================================================
# Serial reader thread
# ============================================================

def serial_reader():
    """Opens the serial port, parses firmware output, drives CSV writing."""
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=2)
        except serial.SerialException as exc:
            message = f"Cannot open {SERIAL_PORT}: {exc}"
            print(f"[SERIAL] {message} — retrying in 3 s", file=sys.stderr)
            with data_lock:
                latest_data["serial_connected"] = False
                latest_data["serial_error"] = message
            time.sleep(3)
            continue

        print(f"[SERIAL] Opened {SERIAL_PORT} at {SERIAL_BAUD} baud")
        with data_lock:
            latest_data["serial_connected"] = True
            latest_data["serial_error"] = None

        prev_total_errors = 0
        pending = []
        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("---"):
                    if pending:
                        parsed = _parse_cycle(pending)
                        if parsed:
                            window_errors = parsed["total_i2c_errors"] - prev_total_errors
                            if window_errors < 0:
                                window_errors = 0
                            prev_total_errors = parsed["total_i2c_errors"]

                            with data_lock:
                                for ax in ("x", "y", "z"):
                                    for feat in ("rms", "peak", "cf", "kurtosis",
                                                 "skewness", "dom_freq", "spectral_entropy"):
                                        latest_data[f"{feat}_{ax}"] = parsed[f"{feat}_{ax}"]
                                latest_data["i2c_errors"] = window_errors
                                latest_data["cycle"] = parsed["cycle"]
                                latest_data["actual_fs"] = parsed["actual_fs"]
                                latest_data["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            if session_state["recording"]:
                                write_row(parsed, window_errors)
                    pending = []
                else:
                    pending.append(line)
        except (serial.SerialException, OSError) as exc:
            message = f"Serial read failed on {SERIAL_PORT}: {exc}"
            print(f"[SERIAL] {message} — reconnecting in 3 s", file=sys.stderr)
            with data_lock:
                latest_data["serial_connected"] = False
                latest_data["serial_error"] = message
        finally:
            try:
                ser.close()
            except Exception:
                pass
            with data_lock:
                latest_data["serial_connected"] = False
            print("[SERIAL] Port closed; retrying.")
        time.sleep(3)


# ============================================================
# Flask app
# ============================================================

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vibration Data Logger</title>
  <style>
    body  { font-family: Arial, sans-serif; margin: 0; padding: 20px;
            background: #111; color: #fff; }
    h1    { text-align: center; margin-bottom: 10px; }
    h2    { color: #aaa; font-size: 14px; margin: 18px 0 6px; text-transform: uppercase; letter-spacing: 1px; }
    .status-bar { text-align: center; padding: 10px; border-radius: 8px;
                  background: #1c1c1c; margin-bottom: 12px; font-size: 14px; }
    .session-bar { background: #141424; border: 1px solid #2a2a4a;
                   border-radius: 8px; padding: 14px; margin-bottom: 14px; }
    .ctrl-row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    select, button, input[type=number] {
                     font-size: 14px; padding: 7px 14px; border-radius: 6px;
                     border: none; cursor: pointer; }
    select, input[type=number] { background: #2a2a2a; color: #fff; }
    input[type=number] { width: 64px; padding: 7px 8px; }
    #btn-start    { background: #2a7a2a; color: #fff; }
    #btn-stop     { background: #7a2a2a; color: #fff; }
    #btn-start:disabled, #btn-stop:disabled { opacity: 0.35; cursor: default; }
    .session-info { margin-top: 10px; font-size: 13px; color: #aaa; }
    .rec-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
               background: #444; margin-right: 5px; vertical-align: middle; }
    .rec-dot.active { background: #e44; animation: blink 1s infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 8px; }
    th  { color: #777; font-weight: normal; text-align: center; padding: 5px 10px;
          border-bottom: 1px solid #2a2a2a; }
    th.feat { text-align: left; }
    td  { padding: 4px 10px; text-align: center; font-variant-numeric: tabular-nums; }
    td.feat-name { text-align: left; color: #bbb; }
    tr:nth-child(even) { background: #181818; }
    .dim { color: #555; font-size: 11px; }
  </style>
</head>
<body>
<h1>Vibration Data Logger</h1>

<div class="status-bar">
  Serial: <strong id="conn-status">WAITING</strong>
  &emsp;|&emsp; Cycle: <span id="cycle">—</span>
  &emsp;|&emsp; Fs: <span id="actual-fs">—</span> Hz
  &emsp;|&emsp; I²C Δerr: <span id="i2c-err">—</span>
  &emsp;|&emsp; <span id="last-update" style="color:#777">—</span>
</div>

<div class="session-bar">
  <h2 style="margin-top:0">Session control</h2>
  <div class="ctrl-row">
    <span style="color:#aaa;font-size:13px;">Fault class:</span>
    <select id="label-select">
      <option>Normal</option>
      <option>Imbalance</option>
      <option>Misalignment</option>
      <option>Looseness</option>
    </select>
    <span style="color:#aaa;font-size:13px;">Duration:</span>
    <input type="number" id="duration-input" min="0.5" step="0.5" placeholder="min"
           title="Recording duration in minutes">
    <span style="color:#777;font-size:12px">min</span>
    <button id="btn-start" onclick="startSession()">Start</button>
    <button id="btn-stop"  onclick="stopSession()" disabled>Stop</button>
  </div>
  <div class="session-info">
    <span class="rec-dot" id="rec-dot"></span>
    <span id="session-status">No session active</span>
  </div>
</div>

<h2>Latest window</h2>
<table>
  <tr>
    <th class="feat">Feature</th>
    <th>X</th><th>Y</th><th>Z</th>
    <th>Unit</th>
  </tr>
  <tr>
    <td class="feat-name">RMS</td>
    <td id="rms_x">—</td><td id="rms_y">—</td><td id="rms_z">—</td>
    <td class="dim">g</td>
  </tr>
  <tr>
    <td class="feat-name">Peak</td>
    <td id="peak_x">—</td><td id="peak_y">—</td><td id="peak_z">—</td>
    <td class="dim">g</td>
  </tr>
  <tr>
    <td class="feat-name">Crest Factor <span class="dim">(display only)</span></td>
    <td id="cf_x">—</td><td id="cf_y">—</td><td id="cf_z">—</td>
    <td class="dim">—</td>
  </tr>
  <tr>
    <td class="feat-name">Kurtosis</td>
    <td id="kurtosis_x">—</td><td id="kurtosis_y">—</td><td id="kurtosis_z">—</td>
    <td class="dim">excess</td>
  </tr>
  <tr>
    <td class="feat-name">Skewness</td>
    <td id="skewness_x">—</td><td id="skewness_y">—</td><td id="skewness_z">—</td>
    <td class="dim">—</td>
  </tr>
  <tr>
    <td class="feat-name">Dom. Frequency</td>
    <td id="dom_freq_x">—</td><td id="dom_freq_y">—</td><td id="dom_freq_z">—</td>
    <td class="dim">Hz</td>
  </tr>
  <tr>
    <td class="feat-name">Spectral Entropy</td>
    <td id="spectral_entropy_x">—</td>
    <td id="spectral_entropy_y">—</td>
    <td id="spectral_entropy_z">—</td>
    <td class="dim">[0,1]</td>
  </tr>
</table>

<script>
const FEATS = [
  "rms_x","rms_y","rms_z",
  "peak_x","peak_y","peak_z",
  "cf_x","cf_y","cf_z",
  "kurtosis_x","kurtosis_y","kurtosis_z",
  "skewness_x","skewness_y","skewness_z",
  "dom_freq_x","dom_freq_y","dom_freq_z",
  "spectral_entropy_x","spectral_entropy_y","spectral_entropy_z"
];

function fmt(v) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(4);
}

async function updateData() {
  try {
    const r = await fetch("/api/data");
    const d = await r.json();
    const cs = document.getElementById("conn-status");
    cs.textContent = d.serial_connected ? "CONNECTED" : "DISCONNECTED";
    cs.style.color = d.serial_connected ? "#4f4" : "#f44";
    document.getElementById("cycle").textContent      = d.cycle ?? "—";
    document.getElementById("actual-fs").textContent  = d.actual_fs != null ? Number(d.actual_fs).toFixed(1) : "—";
    document.getElementById("i2c-err").textContent    = d.i2c_errors ?? "—";
    document.getElementById("last-update").textContent = d.last_update || "—";
    FEATS.forEach(col => {
      const el = document.getElementById(col);
      if (el) el.textContent = fmt(d[col]);
    });
  } catch(e) {
    document.getElementById("conn-status").textContent = "SERVER ERROR";
  }
}

async function updateSession() {
  try {
    const r = await fetch("/api/session/status");
    const s = await r.json();
    const dot  = document.getElementById("rec-dot");
    const info = document.getElementById("session-status");

    if (s.recording) {
      dot.className = "rec-dot active";
      const remStr = s.remaining_s != null
        ? ` — ${Math.floor(s.remaining_s/60)}m ${s.remaining_s%60}s left` : "";
      info.innerHTML = `Recording — ${s.label} &nbsp;|&nbsp; `
        + `<code>${s.session_id}</code> &nbsp;|&nbsp; ${s.row_count} windows${remStr}`;
      document.getElementById("btn-start").disabled    = true;
      document.getElementById("btn-stop").disabled     = false;
      document.getElementById("label-select").disabled = true;
      document.getElementById("duration-input").disabled = true;
    } else {
      dot.className = "rec-dot";
      info.innerHTML = s.session_id
        ? `Done — ${s.label} &nbsp;|&nbsp; ${s.row_count} windows &nbsp;|&nbsp; <code>${s.csv_path}</code>`
        : "No session active";
      document.getElementById("btn-start").disabled    = false;
      document.getElementById("btn-stop").disabled     = true;
      document.getElementById("label-select").disabled = false;
      document.getElementById("duration-input").disabled = false;
    }
  } catch(e) {}
}

async function startSession() {
  const label = document.getElementById("label-select").value;
  const duration_min = parseFloat(document.getElementById("duration-input").value);
  if (!duration_min || duration_min <= 0) {
    alert("Enter a duration in minutes before starting."); return;
  }
  const r = await fetch("/api/session/start", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({label, duration_min})
  });
  const d = await r.json();
  if (!r.ok) { alert("Error: " + d.error); return; }
  updateSession();
}

async function stopSession() {
  const r = await fetch("/api/session/stop", {method: "POST"});
  const d = await r.json();
  if (!r.ok) { alert("Error: " + d.error); return; }
  updateSession();
}

setInterval(updateData,    500);
setInterval(updateSession, 1000);
updateData();
updateSession();
</script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(HTML)


@app.route("/api/data")
def api_data():
    with data_lock:
        return jsonify(dict(latest_data))


@app.route("/api/session/start", methods=["POST"])
def session_start():
    body  = request.get_json(silent=True) or {}
    label = body.get("label", "").strip()
    try:
        duration_min = float(body.get("duration_min", 0))
    except (TypeError, ValueError):
        duration_min = 0.0
    if duration_min <= 0:
        return jsonify({"error": "duration_min must be a positive number of minutes"}), 400

    if label not in VALID_LABELS:
        return jsonify({"error": f"label must be one of {sorted(VALID_LABELS)}"}), 400

    with csv_lock:
        if session_state["recording"]:
            return jsonify({"error": "A session is already recording. Stop it first."}), 400

        session_id = "S" + datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            f, writer, path = _open_session_csv(session_id, label)
        except OSError as exc:
            return jsonify({"error": f"Could not create CSV: {exc}"}), 500

        t = threading.Timer(duration_min * 60, _stop_session_by_timer)
        t.daemon = True

        session_state.update({
            "recording":    True,
            "session_id":   session_id,
            "label":        label,
            "row_count":    0,
            "error_count":  0,
            "csv_path":     path,
            "duration_min": duration_min,
            "deadline":     time.time() + duration_min * 60,
            "_timer":       t,
            "_file":        f,
            "_writer":      writer,
        })
        t.start()

    print(f"[SESSION] Start — {label} / {session_id} / {duration_min}min -> {path}")
    return jsonify({"session_id": session_id, "label": label,
                    "csv_path": path, "duration_min": duration_min})


@app.route("/api/session/stop", methods=["POST"])
def session_stop():
    with csv_lock:
        if not session_state["recording"]:
            return jsonify({"error": "No session is currently recording."}), 400

        if session_state.get("_timer"):
            session_state["_timer"].cancel()
            session_state["_timer"] = None
        session_state["recording"] = False
        session_state["deadline"]  = None
        session_state["_file"].flush()
        session_state["_file"].close()
        session_state["_file"]   = None
        session_state["_writer"] = None

    print(
        f"[SESSION] Stop — {session_state['label']} / {session_state['session_id']}"
        f" — {session_state['row_count']} rows, {session_state['error_count']} write errors"
    )
    return jsonify({
        "session_id":  session_state["session_id"],
        "label":       session_state["label"],
        "row_count":   session_state["row_count"],
        "error_count": session_state["error_count"],
        "csv_path":    session_state["csv_path"],
    })


@app.route("/api/session/status")
def session_status():
    with csv_lock:
        deadline = session_state.get("deadline")
        return jsonify({
            "recording":   session_state["recording"],
            "session_id":  session_state["session_id"],
            "label":       session_state["label"],
            "row_count":   session_state["row_count"],
            "error_count": session_state["error_count"],
            "csv_path":    session_state["csv_path"],
            "duration_min": session_state.get("duration_min"),
            "remaining_s":  max(0, round(deadline - time.time())) if deadline else None,
        })


# ============================================================
# Timer auto-stop
# ============================================================

def _stop_session_by_timer():
    """Auto-close the session when the recording duration elapses."""
    with csv_lock:
        if not session_state["recording"]:
            return
        session_state["recording"] = False
        session_state["deadline"]  = None
        session_state["_timer"]    = None
        try:
            session_state["_file"].flush()
            session_state["_file"].close()
        except Exception as exc:
            print(f"[SESSION] Error closing CSV on timer: {exc}", file=sys.stderr)
        session_state["_file"]   = None
        session_state["_writer"] = None
    print(f"[SESSION] Timer auto-stop — {session_state['label']} / "
          f"{session_state['session_id']} — {session_state['row_count']} rows")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Extend session_state with timer fields (not in the dict literal above
    # so the dict stays readable; initialised here before any thread starts)
    session_state.setdefault("duration_min", None)
    session_state.setdefault("deadline",     None)
    session_state.setdefault("_timer",       None)

    print()
    print("========================================")
    print(" CNC Vibration Data Logger")
    print("========================================")
    print()
    print(f"Serial port   : {SERIAL_PORT}  ({SERIAL_BAUD} baud)")
    print(f"Web dashboard : http://127.0.0.1:{WEB_PORT}")
    print(f"CSV output dir: {DATA_DIR}")
    print()

    threading.Thread(target=serial_reader, daemon=True).start()

    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)
