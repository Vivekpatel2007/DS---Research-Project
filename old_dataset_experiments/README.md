# Baseline Heart Failure Experiments (Original 299-Patient Cohort)

This directory houses the complete historical and methodological progression of experiments conducted on the original **UCI Heart Failure Clinical Records Dataset** (299 patients, 12 clinical features).

> **Reference Publication:**  
> Md. Mamun Ali, Vian S. Al-Doori, Nubogh Mirzah, Asifa Afsari Hemu, Imran Mahmud, Sami Azam, Kusay Faisal Al-tabatabaie, Kawsar Ahmed, Francis M. Bui, Mohammad Ali Moni.  
> *"A machine learning approach for risk factors analysis and survival prediction of Heart Failure patients"*, **Healthcare Analytics**, Volume 3, November 2023, 100182.  
> PDF available in: [`docs/A machine learning approach for risk factors analysis and survival prediction.pdf`](docs/)

---

## Directory Architecture

```
old_dataset_experiments/
├── README.md                                  # This documentation file
├── data/
│   └── heart_failure_clinical_records_dataset.csv   # Original UCI cohort (299 patients, 13 columns)
├── docs/
│   └── A machine learning approach...pdf      # Published paper from Healthcare Analytics (2023)
├── v1_paper_replication/                      # Direct implementation replicating paper claims
│   ├── heart_failure_paper_implementation.ipynb
│   ├── paper_implementation.py
│   ├── generate_and_execute_notebook.py
│   ├── research.ipynb
│   └── results/                               # Reproductions of Fig 2 - Fig 8
├── v2_rigorous_reanalysis/                    # Methodologically audited, leakage-free benchmark
│   ├── heart_failure_rigorous_reanalysis.ipynb
│   └── heart_failure_rigorous_reanalysis_results.xlsx
└── v3_deep_learning_reanalysis/               # Tabular Deep Learning exploration
    └── heart_failure_deep_learning_reanalysis.ipynb
```

---

## Experimental Versions Breakdown

### Version 1: Direct Paper Replication (`v1_paper_replication/`)
- **Objective:** Faithfully reproduce the exact machine learning pipeline and findings described in Ali et al. (2023).
- **Pipeline Components:**
  - **Data Ingestion & Cleaning:** Outlier median imputation / IQR trimming and SMOTE balancing to replicate reported distributions.
  - **Supervised Classification:** Decision Tree (DT), Random Forest (RF), XGBoost (XGB), Gradient Boosting (GB), and Decision Tree Regressor (DTR) using the paper's specific Table 3 hyperparameters.
  - **Explainability & Clustering:** Feature importance ranking via mutual information and Gini impurity (Table 5–7), followed by Gaussian Mixture Model (GMM) clustering projected onto 2D PCA (Section 2.7, Fig. 7).
  - **Survival Modeling:** Kaplan-Meier survival curves and Cox Proportional Hazards hazard ratio estimation (Fig. 8, Table 8).
- **Artifacts:**
  - Fully executed notebook: [`heart_failure_paper_implementation.ipynb`](v1_paper_replication/heart_failure_paper_implementation.ipynb)
  - Standalone script: [`paper_implementation.py`](v1_paper_replication/paper_implementation.py)
  - Generated publication figures: [`results/`](v1_paper_replication/results/) (`fig2_feature_correlation.png` through `fig8_survival_analysis.png`).

---

### Version 2: Rigorous Methodological Re-Analysis (`v2_rigorous_reanalysis/`)
- **Objective:** Scientifically evaluate the paper's reported 97.78% Random Forest accuracy claim and establish an honest, leakage-free baseline.
- **Key Methodological Corrections:**
  - **Leakage Prevention (`time` feature):** Excluded the post-baseline `time` variable (follow-up duration) from death event classification, as it leaks outcome information into retrospective predictors.
  - **Cross-Validation Integrity:** Replaced single-split / SMOTE-leaked preprocessing with strict, repeated stratified 5-fold cross-validation where all scalers and transformations are fitted strictly within training folds.
  - **Metric Realism:** Documented realistic out-of-sample generalization metrics (AUROC, balanced accuracy, F1-score) under clinically valid testing regimes.
- **Artifacts:**
  - Rigorous evaluation notebook: [`heart_failure_rigorous_reanalysis.ipynb`](v2_rigorous_reanalysis/heart_failure_rigorous_reanalysis.ipynb)
  - Tabulated comparative results: [`heart_failure_rigorous_reanalysis_results.xlsx`](v2_rigorous_reanalysis/heart_failure_rigorous_reanalysis_results.xlsx)

---

### Version 3: Deep Learning Re-Analysis (`v3_deep_learning_reanalysis/`)
- **Objective:** Investigate whether a Multi-Layer Perceptron (PyTorch Deep Tabular MLP) provides representational advantages over tree ensembles on small clinical cohorts.
- **Pipeline Highlights:**
  - Tailored small neural network architecture regularized with dropout and weight decay to prevent overfitting on $N=299$.
  - Class-weighted Cross-Entropy loss instead of synthetic SMOTE generation.
  - Out-of-fold probabilistic calibration and comparison with gradient boosted ensembles.
- **Artifacts:**
  - Executed neural pipeline: [`heart_failure_deep_learning_reanalysis.ipynb`](v3_deep_learning_reanalysis/heart_failure_deep_learning_reanalysis.ipynb)

---

## Dataset Schema (`data/heart_failure_clinical_records_dataset.csv`)

| Feature | Type | Unit / Range | Clinical Meaning |
| :--- | :--- | :--- | :--- |
| `age` | Numeric | Years (40–95) | Patient age |
| `anaemia` | Binary | 0 or 1 | Decrease of red blood cells or hemoglobin |
| `creatinine_phosphokinase` | Numeric | mcg/L (23–7861) | Level of the CPK enzyme in blood |
| `diabetes` | Binary | 0 or 1 | Whether patient has diabetes |
| `ejection_fraction` | Numeric | % (14–80) | Percentage of blood leaving heart at each contraction |
| `high_blood_pressure` | Binary | 0 or 1 | Whether patient has hypertension |
| `platelets` | Numeric | kiloplatelets/mL | Platelets in blood |
| `serum_creatinine` | Numeric | mg/dL (0.5–9.4) | Level of serum creatinine in blood |
| `serum_sodium` | Numeric | mEq/L (113–148) | Level of serum sodium in blood |
| `sex` | Binary | 0 = Female, 1 = Male | Patient sex |
| `smoking` | Binary | 0 or 1 | Patient smoking status |
| `time` | Numeric | Days (4–285) | Follow-up observation period |
| **`DEATH_EVENT`** | Binary | 0 = Survived, 1 = Deceased | Primary clinical endpoint / target |
