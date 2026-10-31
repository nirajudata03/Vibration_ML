<<<<<<< HEAD
# Vibration ML

Local vibration monitoring dashboard using a React frontend, Flask API, sensor logger, and saved Random Forest model.

## Project folders

- `frontend/` — React dashboard and Vite development server
- `backend/` — Flask API, training program, and prediction program
- `data/readings/` — CSV sessions written by `bin/data_logger.py`
- `data/uploads/` — CSV files uploaded from the website
- `data/predictions/` — CSV files returned by the model
- `csv files/` — local training recordings
- `results/` — local trained model and training reports
- `backend/data_logger.py` — serial sensor logger

Recordings, uploaded data, generated results, the trained model, logs, and Node packages are excluded from Git. Keep a local copy of `results/vibration_random_forest.pkl` for prediction. To rebuild it, put training CSVs in `csv files/` and run `python backend/train_vibration_rf.py` from this folder.

## Run locally

1. Install Python dependencies once: `python -m pip install -r backend/requirements.txt`
2. Install frontend dependencies once: `cd frontend` then `npm install`
3. Double-click `start_dashboard.bat` from File Explorer, or run it from this folder.
4. Open the local address printed by Vite, usually `http://127.0.0.1:5173`.
5. Click **Activate backend** in the website to start the sensor logger.
6. Use the **N** menu and choose **Stop all servers** when finished.

Connect the ESP32 and close Arduino Serial Monitor or other programs using its COM port before starting the logger. If needed, edit `SERIAL_PORT` in `backend/data_logger.py`. The old `bin/data_logger.py` path remains as a small launcher for this copy.
=======
# Vibration_ML
>>>>>>> 88db6dc71ebebcda0ffff54e0e5705281c0afa70
