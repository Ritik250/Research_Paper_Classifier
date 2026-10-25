"""Dataset loading, conservative cleaning, deduplication, and class balancing."""
from pathlib import Path
import re
import unicodedata
import pandas as pd

LABELS = ["Science (including Engineering)", "Business", "Social Sciences"]
SOURCE_FILES = {
    "Science (including Engineering)": ["sciences.csv", "engineering.csv"],
    "Business": ["business_original.csv", "business_2000.csv"],
    "Social Sciences": ["social_science_original.csv", "social_science_2000.csv"],
}
EXCLUDED_TYPES = {"erratum", "editorial", "letter", "note", "conference review", "retracted"}
BOILERPLATE_PATTERNS = [
    re.compile(r"\b(?:©|copyright)\s*\d{4}[^.\n]*", re.I),
    re.compile(r"\ball rights reserved\b", re.I),
    re.compile(r"\bpublisher(?:'s)? disclaimer\b[^.\n]*", re.I),
]

def normalize_text(value):
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()

def clean_abstract(value):
    text = str(value or "").strip()
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _read(path, label, filename, priority):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for required in ["Abstract", "Title"]:
        if required not in df.columns:
            raise ValueError(f"{filename} missing required column {required!r}")
    for col in ["EID", "DOI", "Link", "Source title", "Document Type", "Year"]:
        if col not in df.columns:
            df[col] = ""
    out = pd.DataFrame({
        "source_file": filename, "source_priority": priority,
        "label": label, "eid": df["EID"].astype(str).str.strip(),
        "doi": df["DOI"].astype(str).str.strip(), "link": df["Link"].astype(str).str.strip(),
        "title": df["Title"].astype(str).str.strip(), "abstract": df["Abstract"].map(clean_abstract),
        "source_title": df["Source title"].astype(str).str.strip(),
        "document_type": df["Document Type"].astype(str).str.strip(),
        "year": df["Year"].astype(str).str.strip(),
    })
    out["abstract_norm"] = out["abstract"].map(normalize_text)
    out["title_norm"] = out["title"].map(normalize_text)
    out["eid_key"] = out["eid"].str.lower().str.strip()
    out["doi_key"] = out["doi"].str.lower().str.strip().str.replace(r"^https?://(?:dx\.)?doi\.org/", "", regex=True)
    out["doc_type_norm"] = out["document_type"].str.lower().str.strip()
    return out

def prepare_dataset(raw_dir, processed_dir):
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    pieces=[]
    # Newer exports take precedence over the earlier exports for repeated same-class records.
    priorities = {"sciences.csv": 10, "engineering.csv": 20, "business_original.csv": 10,
                  "business_2000.csv": 20, "social_science_original.csv": 10, "social_science_2000.csv": 20}
    for label, names in SOURCE_FILES.items():
        for name in names:
            path=raw_dir/name
            if not path.exists():
                raise FileNotFoundError(f"Expected source CSV: {path}")
            pieces.append(_read(path, label, name, priorities[name]))
    all_rows=pd.concat(pieces, ignore_index=True)
    quality=[]
    def record(stage, before, after, detail):
        quality.append({"stage":stage,"records_before":int(before),"records_after":int(after),"removed":int(before-after),"notes":detail})
    record("Raw concatenation",len(all_rows),len(all_rows),"All six supplied CSV files; labels inferred from source filename as requested.")
    # Document type filtering
    before=len(all_rows)
    all_rows=all_rows[~all_rows["doc_type_norm"].isin(EXCLUDED_TYPES)].copy()
    record("Exclude non-research document types",before,len(all_rows),"Excluded document types: "+", ".join(sorted(EXCLUDED_TYPES)))
    before=len(all_rows)
    all_rows=all_rows[all_rows["abstract_norm"].str.len().ge(80) & all_rows["title_norm"].str.len().ge(3)].copy()
    record("Require usable title and abstract",before,len(all_rows),"Abstract must contain at least 80 normalized characters; blank/short abstracts removed.")
    # Identify EIDs mapped to multiple classes and exclude all such EIDs.
    eids=all_rows[all_rows["eid_key"].ne("")].groupby("eid_key")["label"].nunique()
    conflict_eids=set(eids[eids>1].index)
    conflict_rows=all_rows[all_rows["eid_key"].isin(conflict_eids)].copy()
    conflict_rows.to_csv(processed_dir/"conflicting_eids_audit.csv",index=False)
    before=len(all_rows)
    all_rows=all_rows[~all_rows["eid_key"].isin(conflict_eids)].copy()
    record("Exclude cross-label EID conflicts",before,len(all_rows),f"Excluded all rows for {len(conflict_eids)} EIDs associated with more than one provisional class.")
    # Also exclude normalized abstracts assigned to multiple classes.
    abstract_class_counts=all_rows[all_rows["abstract_norm"].ne("")].groupby("abstract_norm")["label"].nunique()
    conflict_abstracts=set(abstract_class_counts[abstract_class_counts>1].index)
    conflict_abs_rows=all_rows[all_rows["abstract_norm"].isin(conflict_abstracts)].copy()
    conflict_abs_rows.to_csv(processed_dir/"conflicting_abstracts_audit.csv",index=False)
    before=len(all_rows)
    all_rows=all_rows[~all_rows["abstract_norm"].isin(conflict_abstracts)].copy()
    record("Exclude cross-label abstract conflicts",before,len(all_rows),f"Excluded all rows for {len(conflict_abstracts)} normalized abstracts mapped to multiple provisional classes.")
    # Prefer newer export and stable identifiers when deduplicating same-class records.
    all_rows=all_rows.sort_values(["source_priority","eid_key","doi_key"],ascending=[False,True,True]).copy()
    before=len(all_rows)
    has_eid=all_rows["eid_key"].ne("")
    dup_eid=all_rows[has_eid].duplicated("eid_key",keep="first")
    remove_indices=set(all_rows[has_eid].index[dup_eid])
    all_rows=all_rows.drop(index=list(remove_indices))
    record("Deduplicate by EID",before,len(all_rows),"Same EID retained once; newer export preferred for same-class duplicates.")
    before=len(all_rows)
    nonempty_doi=all_rows["doi_key"].ne("")
    duplicate_doi=all_rows[nonempty_doi].duplicated("doi_key",keep="first")
    all_rows=all_rows.drop(index=all_rows[nonempty_doi].index[duplicate_doi])
    record("Deduplicate by DOI",before,len(all_rows),"Same DOI retained once after cross-label conflicts were removed.")
    before=len(all_rows)
    nonempty_abs=all_rows["abstract_norm"].ne("")
    duplicate_abs=all_rows[nonempty_abs].duplicated("abstract_norm",keep="first")
    all_rows=all_rows.drop(index=all_rows[nonempty_abs].index[duplicate_abs])
    record("Deduplicate normalized abstracts",before,len(all_rows),"Exact normalized abstracts retained once; cross-label cases already audited and excluded.")
    all_rows=all_rows.sort_values(["label","eid_key"]).reset_index(drop=True)
    final=all_rows[["eid","doi","link","title","abstract","source_title","document_type","year","label","source_file"]].copy()
    final.to_csv(processed_dir/"cleaned_dataset.csv",index=False)
    # Class counts before balancing
    counts=final["label"].value_counts().reindex(LABELS,fill_value=0)
    dist=pd.DataFrame({"label":LABELS,"usable_records":counts.values})
    dist.to_csv(processed_dir/"class_distribution_before_balancing.csv",index=False)
    quality.append({"stage":"Final cleaned dataset","records_before":len(final),"records_after":len(final),"removed":0,"notes":"Final unique usable labeled abstracts before balancing."})
    pd.DataFrame(quality).to_csv(processed_dir/"data_quality_report.csv",index=False)
    # Balanced sample: deterministic downsampling to the minority class count.
    if (counts==0).any():
        raise ValueError(f"At least one class has no usable records after cleaning: {counts.to_dict()}")
    n=int(counts.min())
    balanced=(final.groupby("label",group_keys=False).sample(n=n,random_state=42)
              .sample(frac=1,random_state=42).reset_index(drop=True))
    balanced.to_csv(processed_dir/"balanced_dataset.csv",index=False)
    pd.DataFrame({"label":LABELS,"balanced_records":[n]*len(LABELS)}).to_csv(processed_dir/"balanced_class_distribution.csv",index=False)
    dictionary='''# Data dictionary\n\n`cleaned_dataset.csv` contains unique eligible records before balancing. `balanced_dataset.csv` is the deterministic, equal-class sample used for the primary model experiment.\n\n| Column | Meaning |\n|---|---|\n| `eid` | Scopus EID when supplied |\n| `doi` | DOI when supplied |\n| `link` | Source record URL when supplied |\n| `title` | Publication title |\n| `abstract` | Cleaned abstract text used for classification |\n| `source_title` | Journal or publication source title |\n| `document_type` | Source document type |\n| `year` | Publication year as supplied |\n| `label` | Provisional class based on supplied CSV filename |\n| `source_file` | CSV that contributed the retained row |\n\n## Label mapping\n- `sciences.csv` and `engineering.csv` → **Science (including Engineering)**\n- Business source CSVs → **Business**\n- Social Sciences source CSVs → **Social Sciences**\n\nThe filename-based labels are provisional. The original Scopus subject-area filters have not been independently verified. Cross-class EID or normalized-abstract conflicts are excluded and recorded in audit CSVs. The balanced dataset is downsampled without replacement; no records are synthesized or duplicated.\n'''
    (processed_dir/"data_dictionary.md").write_text(dictionary,encoding="utf-8")
    return final, balanced, pd.DataFrame(quality), dist
