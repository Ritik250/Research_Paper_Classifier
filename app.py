"""ResearchField AI — interactive research abstract classification."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.predict import load_model, predict_one, predict_batch, LABELS

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="ResearchField AI",page_icon="📚",layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1450px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#14213d,#1d4e89); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin-bottom:.25rem;} .hero p {color:#e6eefb; font-size:1.05rem;}
.small-muted {color: #777; font-size: .9rem;}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📚 ResearchField AI</h1><p>Classify academic research abstracts using explainable machine-learning pipelines.</p></div>',unsafe_allow_html=True)

@st.cache_resource
def get_model(): return load_model()
@st.cache_data
def read_csv(path):
    p=Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

try:
    model=get_model()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.info("Run `python -m src.train` from the project folder to create the model.")
    st.stop()
metadata_path=ROOT/"results/run_metadata.json"
metadata=json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
clean_path=ROOT/"data/processed/cleaned_dataset.csv"
balanced_path=ROOT/"data/processed/balanced_dataset.csv"
clean=pd.read_csv(clean_path) if clean_path.exists() else pd.DataFrame()
balanced=pd.read_csv(balanced_path) if balanced_path.exists() else pd.DataFrame()
comparison=read_csv(ROOT/"results/model_comparison.csv")
report=read_csv(ROOT/"results/classification_report.csv")

tab_overview,tab_single,tab_batch,tab_method=st.tabs(["Overview & Results","Single Abstract","Batch CSV","Methodology"])
with tab_overview:
    st.subheader("Project overview")
    st.write("ResearchField AI predicts one of three provisional research fields from an academic abstract. Labels are inherited from the supplied CSV filenames; original Scopus subject-area filters have not been independently verified.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Unique cleaned records",f"{len(clean):,}" if len(clean) else "—")
    c2.metric("Balanced experiment",f"{len(balanced):,}" if len(balanced) else "—")
    c3.metric("Records per class",f"{int(len(balanced)/3):,}" if len(balanced) else "—")
    c4.metric("Selected model",metadata.get("selected_model","—"))
    if metadata.get("validation_macro_f1") is not None: st.metric("Validation macro F1",f"{metadata['validation_macro_f1']:.1%}")
    st.markdown("#### Class balance")
    if not balanced.empty:
        counts=balanced.label.value_counts().reindex(LABELS,fill_value=0).rename_axis("Class").reset_index(name="Records")
        st.bar_chart(counts.set_index("Class"))
        st.dataframe(counts,use_container_width=True,hide_index=True)
    st.markdown("#### Model comparison")
    if not comparison.empty:
        display_cols=[c for c in ["model","validation_accuracy","validation_macro_precision","validation_macro_recall","validation_macro_f1","validation_weighted_f1","test_accuracy","test_macro_f1"] if c in comparison.columns]
        st.dataframe(comparison[display_cols].style.format({c:"{:.1%}" for c in display_cols if c!="model"}),use_container_width=True,hide_index=True)
        st.caption("Models are selected using validation macro F1. Held-out test metrics are reported only for the selected model in this run; blanks for other models mean they were not evaluated on the held-out test set.")
    cm_path=ROOT/"results/confusion_matrix.png"
    if cm_path.exists():
        st.markdown("#### Held-out test confusion matrix")
        st.image(str(cm_path),use_container_width=True)
    if not report.empty:
        st.markdown("#### Selected model test classification report")
        st.dataframe(report,use_container_width=True)

with tab_single:
    st.subheader("Individual abstract reader")
    st.write("Paste an abstract to predict its likely research field. Use the complete abstract where possible.")
    abstract=st.text_area("Research abstract",height=250,placeholder="Paste the title or abstract text here…")
    st.caption(f"{len(abstract.strip())} characters entered; at least 40 characters are required.")
    if st.button("Classify abstract",type="primary",key="single_predict"):
        try:
            result=predict_one(model,abstract)
            st.success(f"Predicted field: **{result['predicted_class']}**")
            st.info(result["probability_note"])
        except ValueError as exc: st.warning(str(exc))
        except Exception as exc: st.error(f"Prediction failed: {exc}")

with tab_batch:
    st.subheader("Batch classification from CSV")
    st.write("Upload a CSV containing an abstract column. Original columns are retained, with prediction and status columns appended.")
    upload=st.file_uploader("Upload CSV",type=["csv"],key="batch_upload")
    if upload is not None:
        try:
            frame=pd.read_csv(upload)
            if frame.empty: st.warning("The uploaded CSV has no rows.")
            else:
                st.write("Preview"); st.dataframe(frame.head(10),use_container_width=True)
                col=st.selectbox("Choose the abstract column",options=list(frame.columns),index=(list(frame.columns).index("Abstract") if "Abstract" in frame.columns else 0))
                if st.button("Predict batch",type="primary",key="batch_predict"):
                    try:
                        output=predict_batch(model,frame,col)
                        st.session_state["batch_output"]=output
                    except Exception as exc: st.error(f"Batch prediction failed: {exc}")
        except Exception as exc: st.error(f"Could not read the CSV: {exc}")
    if "batch_output" in st.session_state:
        output=st.session_state["batch_output"]
        st.markdown("#### Prediction preview")
        st.dataframe(output.head(30),use_container_width=True)
        valid=output[output["prediction_status"]=="Predicted"]
        skipped=len(output)-len(valid)
        m1,m2,m3=st.columns(3); m1.metric("Input rows",len(output)); m2.metric("Predicted",len(valid)); m3.metric("Skipped",skipped)
        if not valid.empty:
            counts=valid.predicted_class.value_counts().reindex(LABELS,fill_value=0).rename_axis("Predicted class").reset_index(name="Count")
            st.markdown("#### Prediction summary"); st.dataframe(counts,use_container_width=True,hide_index=True)
            st.bar_chart(counts.set_index("Predicted class"))
        st.download_button("Download predictions CSV",output.to_csv(index=False).encode("utf-8-sig"),file_name="researchfield_predictions.csv",mime="text/csv")

with tab_method:
    st.subheader("Methodology")
    st.markdown("""
**Primary classes**
- Science (including Engineering)
- Business
- Social Sciences

**Workflow**
1. Validate the six supplied source CSVs.
2. Exclude selected non-research document types and empty/short abstracts.
3. Audit and exclude cross-class EID conflicts and deduplicate by EID, DOI, and normalized abstract.
4. Create a balanced dataset by deterministic downsampling without replacement.
5. Apply NLTK tokenization, English stopword filtering and Snowball stemming inside the fitted prediction pipeline; then fit TF-IDF on training text only.
6. Compare Logistic Regression, Decision Tree, Multinomial Naive Bayes, kNN, Linear SVM and Random Forest.
7. Select by validation macro F1; evaluate the selected model once on a held-out test set.

**Limitations**
- Filename-based source labels are provisional and may contain label noise.
- A balanced dataset can help compare classes fairly but does not guarantee better predictions on new Scopus exports.
- `LinearSVC` does not provide calibrated probabilities, so no confidence percentage is displayed.
- Academic disciplines overlap in vocabulary; review predictions critically.
""")
    st.markdown("#### Dataset audit files")
    for fname in ["data_quality_report.csv","class_distribution_before_balancing.csv","balanced_class_distribution.csv","conflicting_eids_audit.csv","conflicting_abstracts_audit.csv"]:
        p=ROOT/"data/processed"/fname
        if p.exists(): st.download_button(f"Download {fname}",p.read_bytes(),file_name=fname,mime="text/csv",key=f"download_{fname}")
