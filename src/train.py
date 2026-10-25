"""Train, compare, evaluate, and save six academic abstract classifiers."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from src.nlp_preprocessing import preprocess_texts

SEED = 42
LABELS = ["Science (including Engineering)", "Business", "Social Sciences"]
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=250, min_samples_leaf=2, class_weight="balanced", random_state=SEED),
    "Naive Bayes": MultinomialNB(alpha=0.5),
    "kNN": KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine", algorithm="brute"),
    "SVM (LinearSVC)": LinearSVC(class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=12, max_depth=25, min_samples_leaf=3, max_features=0.2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
}

def make_pipeline(classifier):
    return Pipeline([
        ("nltk_preprocess", FunctionTransformer(preprocess_texts, validate=False)),
        ("tfidf", TfidfVectorizer(lowercase=False, strip_accents="unicode", ngram_range=(1, 2),
            min_df=2, max_df=0.98, max_features=30000, sublinear_tf=True)),
        ("classifier", classifier),
    ])

def evaluate(dataset_path=None, output_dir=None):
    root=Path(__file__).resolve().parents[1]
    dataset_path=Path(dataset_path) if dataset_path else root/"data/processed/balanced_dataset.csv"
    output_dir=Path(output_dir) if output_dir else root/"results"
    output_dir.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(dataset_path).dropna(subset=["abstract","label"]).copy()
    data["abstract"]=data["abstract"].astype(str).str.strip(); data["label"]=data["label"].astype(str).str.strip()
    data=data[data["abstract"].str.len().ge(40)&data["label"].isin(LABELS)].copy()
    if len(data)<1000 or set(data.label.unique())!=set(LABELS):
        raise ValueError("Expected at least 1,000 usable abstracts across all three classes.")
    counts=data.label.value_counts()
    if counts.nunique()!=1:
        raise ValueError(f"Primary balanced experiment expects equal class counts; got {counts.to_dict()}")
    X,y=data["abstract"],data["label"]
    X_trainval,X_test,y_trainval,y_test=train_test_split(X,y,test_size=.15,random_state=SEED,stratify=y)
    X_train,X_val,y_train,y_val=train_test_split(X_trainval,y_trainval,test_size=.1764705882,random_state=SEED,stratify=y_trainval)
    rows=[]; val_scores={}; val_class_metrics={}
    for name,clf in MODELS.items():
        print(f"Training {name} with NLTK + TF-IDF...",flush=True)
        pipe=make_pipeline(clf); pipe.fit(X_train,y_train); pred=pipe.predict(X_val)
        p,r,f,_=precision_recall_fscore_support(y_val,pred,average="macro",zero_division=0)
        _,_,wf,_=precision_recall_fscore_support(y_val,pred,average="weighted",zero_division=0)
        val_scores[name]=float(f)
        report=classification_report(y_val,pred,labels=LABELS,output_dict=True,zero_division=0)
        val_class_metrics[name]={label:{"precision":report[label]["precision"],"recall":report[label]["recall"],"f1-score":report[label]["f1-score"],"support":report[label]["support"]} for label in LABELS}
        rows.append({"model":name,"validation_accuracy":accuracy_score(y_val,pred),"validation_macro_precision":p,
            "validation_macro_recall":r,"validation_macro_f1":f,"validation_weighted_f1":wf})
    best_name=max(val_scores,key=val_scores.get)
    # Refit only the validation-selected model on train+validation, then evaluate once on held-out test.
    # This keeps runtime reasonable; model selection remains strictly validation-based.
    best_pipe=make_pipeline(MODELS[best_name])
    print(f"Refitting selected model {best_name} on train+validation...", flush=True)
    best_pipe.fit(X_trainval,y_trainval)
    best_pred=best_pipe.predict(X_test)
    p,r,f,_=precision_recall_fscore_support(y_test,best_pred,average="macro",zero_division=0)
    _,_,wf,_=precision_recall_fscore_support(y_test,best_pred,average="weighted",zero_division=0)
    results=[]
    for row in rows:
        row.update({"test_accuracy":None,"test_macro_precision":None,"test_macro_recall":None,"test_macro_f1":None,"test_weighted_f1":None})
        if row["model"]==best_name:
            row.update({"test_accuracy":accuracy_score(y_test,best_pred),"test_macro_precision":p,"test_macro_recall":r,
                        "test_macro_f1":f,"test_weighted_f1":wf})
        results.append(row)
    comparison=pd.DataFrame(results).sort_values("validation_macro_f1",ascending=False)
    comparison.to_csv(output_dir/"model_comparison.csv",index=False)
    pd.DataFrame(val_class_metrics[best_name]).transpose().to_csv(output_dir/"selected_model_validation_per_class.csv")
    report=classification_report(y_test,best_pred,labels=LABELS,output_dict=True,zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_dir/"classification_report.csv")
    cm=confusion_matrix(y_test,best_pred,labels=LABELS); cm_df=pd.DataFrame(cm,index=LABELS,columns=LABELS)
    cm_df.to_csv(output_dir/"confusion_matrix.csv")
    plt.figure(figsize=(9,6)); sns.heatmap(cm_df,annot=True,fmt="d",cmap="Blues")
    plt.title(f"Confusion matrix — {best_name} (held-out test set)"); plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.xticks(rotation=25,ha="right"); plt.yticks(rotation=0); plt.tight_layout(); plt.savefig(output_dir/"confusion_matrix.png",dpi=180); plt.close()
    dist=data.label.value_counts().reindex(LABELS)
    plt.figure(figsize=(8,4)); dist.plot(kind="bar"); plt.title("Balanced training dataset class distribution"); plt.xlabel("Class"); plt.ylabel("Records")
    plt.xticks(rotation=20,ha="right"); plt.tight_layout(); plt.savefig(output_dir/"class_distribution.png",dpi=180); plt.close()
    plot_data=comparison.sort_values("validation_macro_f1")
    plt.figure(figsize=(9,5)); plt.barh(plot_data.model,plot_data.validation_macro_f1); plt.xlim(0,1); plt.xlabel("Validation macro F1"); plt.title("Validation model comparison")
    plt.tight_layout(); plt.savefig(output_dir/"validation_macro_f1.png",dpi=180); plt.close()
    model_path=root/"models/research_classifier.joblib"; model_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(best_pipe,model_path,compress=3)
    meta={"selected_model":best_name,"selection_metric":"validation_macro_f1","validation_macro_f1":val_scores[best_name],
          "random_seed":SEED,"dataset":"data/processed/balanced_dataset.csv","n_total_balanced":len(data),
          "class_counts":{k:int(v) for k,v in counts.items()},"n_train":len(X_train),"n_validation":len(X_val),
          "n_train_validation":len(X_trainval),"n_test":len(X_test),"classes":LABELS,
          "text_preprocessing":"NLTK wordpunct_tokenize + SnowballStemmer + English stopword filtering; TF-IDF unigrams/bigrams",
          "model_path":"models/research_classifier.joblib","note":"Model selection uses validation macro F1 only. Held-out test metrics are reported for the selected model only; test metrics are not used to select the model."}
    (output_dir/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(f"Selected model: {best_name}; validation macro F1={val_scores[best_name]:.4f}")
    print(f"Saved model: {model_path}")
    return meta,comparison

if __name__=="__main__":
    meta,comparison=evaluate(); print(json.dumps(meta,indent=2)); print(comparison.to_string(index=False))
