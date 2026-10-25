from pathlib import Path
import pandas as pd
import pytest
from src.predict import load_model, predict_one, predict_batch, LABELS

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope="module")
def model():
    return load_model(ROOT/"models/research_classifier.joblib")

def test_saved_model_loads(model):
    assert hasattr(model,"predict")
    assert "nltk_preprocess" in model.named_steps
    assert "tfidf" in model.named_steps

def test_single_prediction_returns_known_label(model):
    text="This study examines consumer purchase intentions and brand loyalty in online retail markets."
    result=predict_one(model,text)
    assert result["predicted_class"] in LABELS
    assert result["probability"] is None

def test_empty_single_input_rejected(model):
    with pytest.raises(ValueError): predict_one(model,"")

def test_short_single_input_rejected(model):
    with pytest.raises(ValueError): predict_one(model,"too short")

def test_batch_prediction_integrity(model):
    frame=pd.DataFrame({"Abstract":["This study examines consumer purchase intentions and brand loyalty in online retail markets.",""] ,"id":[1,2]})
    result=predict_batch(model,frame,"Abstract")
    assert len(result)==2
    assert "predicted_class" in result and "prediction_status" in result
    assert result.loc[0,"predicted_class"] in LABELS
    assert result.loc[1,"prediction_status"].startswith("Skipped")
    assert result["id"].tolist()==[1,2]

def test_missing_batch_column_rejected(model):
    with pytest.raises(ValueError): predict_batch(model,pd.DataFrame({"text":["x"]}),"Abstract")

def test_balanced_dataset_has_equal_classes():
    df=pd.read_csv(ROOT/"data/processed/balanced_dataset.csv")
    assert df["label"].value_counts().nunique()==1
    assert len(df)>=1000
