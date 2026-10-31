Vibration_ML CSV folders

- data/readings/    New CSV readings saved by backend/data_logger.py
- data/uploads/     CSV files selected through the website
- data/predictions/ Prediction CSV files created by the model or website
- csv files/        Local recordings used to train the model
- results/          Local trained model and generated reports (keep as-is)

Recordings, uploads, predictions, and trained model files stay on this computer and are not committed to Git.

Example prediction command from the Vibration_ML project folder:
python backend\predict_vibration_fault.py data\readings\your_recording.csv --output data\predictions\your_recording_predictions.csv
