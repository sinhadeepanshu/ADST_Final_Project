# AI Predictive Maintenance - Machine Failure Prediction

## What this project does
Predicts machine-failure risk using the AI4I 2020 tabular dataset.

## Current implementation
- EDA-ready dataset
- Data preprocessing
- Logistic Regression and Random Forest comparison
- Failure classification
- Anomaly detection with Isolation Forest
- Transparent 0-100 risk-derived health score
- Streamlit dashboard
- GitHub/Streamlit deployment ready

## Best model from this run
Random Forest

See `model_results.csv` for metrics.

## Important limitation
The project report mentions RUL and 24/48/72-hour forecasts. The supplied AI4I 2020 CSV is static tabular data without the required lifecycle/time-to-event target and timestamp sequence. Therefore this implementation does NOT falsely claim a true RUL or 24/48/72-hour forecast. Those require a suitable time-series/lifecycle dataset.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Create a GitHub repository.
2. Upload all files/folders.
3. Open Streamlit Community Cloud.
4. Choose the GitHub repository and `app.py`.
5. Deploy.

## Files
- `app.py` - Streamlit web application
- `train_model.py` - retrain model
- `model.pkl` - trained model
- `anomaly_model.pkl` - anomaly detector
- `requirements.txt` - dependencies
- `data/ai4i2020.csv` - dataset
- `model_results.csv` - model comparison
