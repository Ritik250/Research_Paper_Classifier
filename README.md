# ResearchField AI — Balanced Research Abstract Classification

ResearchField AI predicts the likely field of an academic publication from its abstract. The primary experiment has three classes:
Check out the [Google Homepage]([https://www.google.com](https://researchpaperclassifier-t3wzhia9ychfgnv2srnzbk.streamlit.app/)).


1. **Science (including Engineering)**
2. **Business**
3. **Social Sciences**

The project uses conventional Python tooling: pandas, scikit-learn, NLTK, TF-IDF, Joblib, matplotlib/seaborn, pytest, and Streamlit.

## Current dataset and cleaning results

The project uses six supplied Scopus-export CSVs: the original Science and Engineering files, the original Business and Social Sciences files, plus the new 2,000-row Business and Social Sciences exports. Filename-based labels are provisional; the original Scopus subject-area filters were not independently verified.

| Stage | Records |
|---|---:|
| Rows concatenated from six files | 8,000 |
| After excluded document types | 7,952 |
| After requiring usable title/abstract | 7,926 |
| After excluding EIDs with conflicting provisional labels | 4,998 |
| After EID deduplication | 3,231 |
| Final unique cleaned dataset | **3,231** |
| Balanced primary experiment | **2,526** |
| Balanced records per class | **842** |

### Final cleaned class counts before balancing

| Class | Usable unique records |
|---|---:|
| Science (including Engineering) | 842 |
| Business | 1,185 |
| Social Sciences | 1,204 |

The balanced dataset is created by deterministic downsampling without replacement to 842 records per class. No synthetic abstracts are created. The full cleaned dataset remains available in `data/processed/cleaned_dataset.csv`.

### Important data-quality findings

- The Science and Engineering files share the same 1,000 EIDs. Those records are not counted twice.
- The supplied Business and Social Sciences sources have cross-label EID overlap. The preparation script excludes all rows for an EID assigned to more than one provisional class, and saves the affected records to `conflicting_eids_audit.csv`.
- Errata, editorials, letters, notes, conference reviews, and retracted items are excluded. Records with missing or very short normalized abstracts are also excluded.
- Duplicate EIDs, DOIs, and normalized abstracts are removed conservatively.
- See `data/processed/data_quality_report.csv`, `conflicting_eids_audit.csv`, and `data_dictionary.md` for the audit trail.

## Model and actual measured results

Text preprocessing is part of the saved scikit-learn pipeline:

1. NLTK `wordpunct_tokenize` tokenization.
2. Lowercasing, URL/email removal, English stopword filtering, and NLTK Snowball stemming.
3. TF-IDF with unigrams and bigrams, fitted only on training text.
4. One of six classifiers: Logistic Regression, Decision Tree, Multinomial Naive Bayes, kNN, Linear SVM, or Random Forest.

The data is split into train, validation, and held-out test partitions using stratification and random seed 42. Model selection is based on **validation macro F1**, not test metrics.

| Model | Validation accuracy | Validation macro F1 | Held-out test accuracy | Held-out test macro F1 |
|---|---:|---:|---:|---:|
| Linear SVM | 82.85% | **82.60%** | 78.89% | 78.69% |
| Logistic Regression | 82.32% | 82.13% | Not evaluated separately | Not evaluated separately |
| Random Forest | 79.68% | 79.39% | Not evaluated separately | Not evaluated separately |
| Naive Bayes | 79.42% | 79.09% | Not evaluated separately | Not evaluated separately |
| kNN | 78.63% | 78.32% | Not evaluated separately | Not evaluated separately |
| Decision Tree | 66.23% | 66.33% | Not evaluated separately | Not evaluated separately |

The validation-selected model is **Linear SVM (`LinearSVC`)**. It achieved 78.89% accuracy and 78.69% macro F1 on the held-out test set. Its test-set per-class metrics are in `results/classification_report.csv`.

| Test class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Science (including Engineering) | 76.82% | 92.06% | 83.75% |
| Business | 85.71% | 71.43% | 77.92% |
| Social Sciences | 75.61% | 73.23% | 74.40% |

Balancing improves comparability between classes but does not guarantee improved generalization to every external Scopus export. The 20-abstract external Business batch has not been included in training or model selection, and this rebuilt version has not yet been evaluated against that batch in this run.

## Project structure

```text
researchfield_ai_v2/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                         # Unmodified copies of the six supplied CSVs
│   └── processed/
│       ├── cleaned_dataset.csv      # All eligible unique records before balancing
│       ├── balanced_dataset.csv     # Balanced primary experiment dataset
│       ├── data_quality_report.csv
│       ├── data_dictionary.md
│       ├── class_distribution_before_balancing.csv
│       ├── balanced_class_distribution.csv
│       ├── conflicting_eids_audit.csv
│       └── conflicting_abstracts_audit.csv
├── models/research_classifier.joblib
├── notebooks/research_classification_colab.ipynb
├── results/
│   ├── model_comparison.csv
│   ├── classification_report.csv
│   ├── selected_model_validation_per_class.csv
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   ├── validation_macro_f1.png
│   └── run_metadata.json
├── src/
│   ├── data_prep.py
│   ├── nlp_preprocessing.py
│   ├── predict.py
│   └── train.py
└── tests/test_prediction.py
```

## Run locally on Windows

Use Python 3.13 if available. Open a terminal in the project root (the folder containing `app.py` and `requirements.txt`).

```cmd
py -3.13 -m venv .venv313
.venv313\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the environment already exists, just activate it:

```cmd
.venv313\Scripts\activate.bat
```

### Rebuild the cleaned and balanced datasets

```cmd
python -c "from src.data_prep import prepare_dataset; prepare_dataset('data/raw','data/processed')"
```

### Retrain and evaluate models

```cmd
python -m src.train
```

This creates/overwrites the saved pipeline and result files. Back up the existing `models/` and `results/` folders first if you need to retain the current run.

### Launch Streamlit

```cmd
python -m streamlit run app.py
```

Open the local URL shown in the terminal, normally `http://localhost:8501`.

### Run tests

```cmd
python -m pytest -q
```

## Google Colab

Open `notebooks/research_classification_colab.ipynb` in Google Colab. Upload the project ZIP when prompted, then run the cells in order. The notebook installs dependencies, extracts the project, reruns preparation/training, displays actual results, and offers the rebuilt project as a ZIP download.

## Streamlit features

- Project overview, balanced dataset statistics, selected model and evaluation results.
- Individual abstract input with validation and predicted field.
- Batch CSV upload, abstract-column selection, preview, skipped-row handling, class summary, and downloadable predictions.
- Methodology, limitations, and downloadable data-quality audits.
- Uses the saved pipeline for both prediction modes; does not retrain during app startup.
- Does not display a probability/confidence percentage for `LinearSVC`, which does not expose calibrated probabilities through `predict_proba`.

## Limitations and responsible interpretation

- Source CSV filenames are used as provisional labels. Subject-area filters were not independently verified.
- Cross-label conflicts are excluded rather than arbitrarily relabeled; this removes a substantial number of overlapping EIDs and should be discussed in a viva.
- Science/Engineering source files and old/new exports contain repeated records; audit files document the treatment.
- Academic fields overlap in vocabulary. Predictions should be treated as decision support, not ground truth.
- Class balancing down-samples the two larger classes, which can discard some diversity. The full cleaned dataset is preserved to support alternative experiments.
- Linear SVM's decision boundary is not a calibrated probability. No percentage is presented as confidence.
- The local environment used to create this run had Python 3.13, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, NLTK 3.9.2, and Joblib 1.5.3. Streamlit's live browser UI still needs to be launched and smoke-tested on the target laptop after dependencies are installed.

## Troubleshooting

- **NumPy attempts to compile from source:** use a Python version with compatible wheels (Python 3.13 is the tested target), create a fresh virtual environment, and reinstall requirements.
- **Model cannot be loaded:** run commands from the project root and verify `models/research_classifier.joblib` exists. Retrain with `python -m src.train` if needed.
- **`ModuleNotFoundError: src`:** ensure the current working directory is the project root and run `python -m src.train` rather than executing `src/train.py` directly.
- **NLTK corpus download prompt:** this implementation does not require downloading NLTK corpora; it uses NLTK tokenization and Snowball stemming with scikit-learn's English stop-word list.
- **Poor predictions on a new Scopus batch:** inspect the actual abstract text, source distribution and label provenance; don't tune against the final held-out test set or simply force a class prediction.
