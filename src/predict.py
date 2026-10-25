"""Shared prediction functions for the notebook, tests, and Streamlit app."""
from pathlib import Path
import joblib
import pandas as pd

LABELS=["Science (including Engineering)","Business","Social Sciences"]

def load_model(model_path=None):
    root=Path(__file__).resolve().parents[1]
    path=Path(model_path) if model_path else root/"models/research_classifier.joblib"
    if not path.exists(): raise FileNotFoundError(f"Saved model not found: {path}. Run python -m src.train first.")
    return joblib.load(path)

def predict_one(model, abstract):
    if not isinstance(abstract,str) or not abstract.strip(): raise ValueError("Please enter a non-empty abstract.")
    if len(abstract.strip())<40: raise ValueError("Abstract is too short. Enter at least 40 characters.")
    pred=str(model.predict([abstract.strip()])[0])
    result={"predicted_class":pred,"probability":None,"probability_note":"This model does not provide calibrated class probabilities; prediction is a class label, not a confidence percentage."}
    if hasattr(model,"predict_proba"):
        try:
            probs=model.predict_proba([abstract.strip()])[0]
            result["probability"]=float(max(probs)); result["probability_note"]="Maximum model probability; not necessarily calibrated confidence."
        except Exception: pass
    return result

def predict_batch(model, frame, abstract_column):
    if not isinstance(frame,pd.DataFrame): raise TypeError("Input must be a pandas DataFrame.")
    if abstract_column not in frame.columns: raise ValueError(f"Column {abstract_column!r} not found in CSV.")
    out=frame.copy()
    texts=out[abstract_column].fillna("").astype(str).str.strip()
    valid=texts.str.len().ge(40)
    predictions=[]; notes=[]
    if valid.any(): predictions=pd.Series(model.predict(texts[valid].tolist()),index=texts[valid].index)
    out["predicted_class"]="Not predicted: empty/short abstract"
    if valid.any(): out.loc[valid,"predicted_class"]=predictions
    out["prediction_status"]="Predicted"; out.loc[~valid,"prediction_status"]="Skipped: empty/short abstract"
    return out
