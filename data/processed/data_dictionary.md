# Data dictionary

`cleaned_dataset.csv` contains unique eligible records before balancing. `balanced_dataset.csv` is the deterministic, equal-class sample used for the primary model experiment.

| Column | Meaning |
|---|---|
| `eid` | Scopus EID when supplied |
| `doi` | DOI when supplied |
| `link` | Source record URL when supplied |
| `title` | Publication title |
| `abstract` | Cleaned abstract text used for classification |
| `source_title` | Journal or publication source title |
| `document_type` | Source document type |
| `year` | Publication year as supplied |
| `label` | Provisional class based on supplied CSV filename |
| `source_file` | CSV that contributed the retained row |

## Label mapping
- `sciences.csv` and `engineering.csv` → **Science (including Engineering)**
- Business source CSVs → **Business**
- Social Sciences source CSVs → **Social Sciences**

The filename-based labels are provisional. The original Scopus subject-area filters have not been independently verified. Cross-class EID or normalized-abstract conflicts are excluded and recorded in audit CSVs. The balanced dataset is downsampled without replacement; no records are synthesized or duplicated.
