# ehr-lung-disease-prediction

## 1. Overview
This project extracts a patient cohort from Synthea‑generated synthetic EHRs and builds ML models to predict lung disease. It covers:
   
   1. **Cohort Extraction**  
      - Flag patients with lung conditions via SNOMED CT codes & keyword matching.  
      - Filter to adults (≥ 18 years) with ≥ 1 inpatient encounter.  
   2. **Feature Engineering**  
      - Demographics: age at first encounter, current/death age, sex, race/ethnicity.  
      - Comorbidity count.  
      - Vital/sign observations in first 48 h (mean/min/max).  
   3. **Preprocessing & Modeling**  
      - Impute & scale numeric features; one‑hot encode categoricals.  
      - Binary classification (disease vs. none) & multiclass (none/pneumonia/asthma/COPD).  
      - Algorithms: Logistic Regression, Random Forest, XGBoost.  
      - Hyperparameter tuning with GridSearchCV (5‑fold CV).  
   4. **Evaluation**  
      - Metrics: AUROC, AUPRC, sensitivity, specificity, F1‑score.  
      - Confusion matrices, ROC & PR curves.  
   5. **Model Persistence**  
      - Best pipelines saved via `joblib` for later inference.

---

## 2. Repository Structure
   ```
   ehr-lung-disease-prediction/
   ├── data/
   │   ├── patients.csv
   │   ├── conditions.csv
   │   ├── encounters.csv
   │   └── observations.csv
   ├── notebook.ipynb
   ├── models/
   │   ├── best_binary_xgb_pipeline.joblib
   │   └── best_multiclass_xgb_pipeline.joblib
   ├── requirements.txt
   └── README.md
   ```

---

## 3. Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Kanai204/ehr-lung-disease-prediction.git
   cd ehr-lung-disease-prediction
2. **Create an environment & install dependencies**
   ```
   python3 -m venv venv
   source venv/bin/activate       # Linux/Mac
   .\venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
3. **Install dependencies**
   Install dependencies
4. **Data**
   Download the Synthea CSV files from the Harvard Dataverse:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BWDKXS

   Place the following four files into the `data/` folder:

---

## 4. Running the Analysis
1. **Launch Jupyter:**
   ```bash
    jupyter notebook notebook.ipynb
2. Run each cell in order, or click Kernel ▶ Restart & Run All.

---

## 5. Results
Binary lung‑disease prediction
– Best model: XGBoost (CV ROC‑AUC ≈ 0.728).

Multiclass disease‑type (none/pneumonia/asthma/COPD)
– Best model: XGBoost (CV accuracy ≈ 0.827, macro F1 ≈ 0.368).

Saved pipelines live in `models/`.

---

## 6.Requirements
```
pandas
numpy
scikit-learn
xgboost
matplotlib
jupyter
joblib
```

---

## 7. References
- Overhage, J. “Synthea: Synthetic Patient Population Generator,” 2019.

- Krumholz, H. M. et al., “Modeling EHR‑based Predictive Algorithms,” J. Biomed. Inform., 64, pp. 33–41, 2017.
