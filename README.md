# Bahraini Currency Detector

Detects Bahraini banknotes (0.5, 1, 5, 10 and 20 BHD) in a photo using a fine-tuned YOLO model.

**Live app:** [add Streamlit link here]

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – Streamlit app (upload a photo, see detections)
- `best.pt` – fine-tuned YOLO weights
- `Bahraini_Currency_Detection.ipynb` – data download, fine-tuning and results

## Data
Roboflow project `bahrain_currency_object_detector` (version 2).
