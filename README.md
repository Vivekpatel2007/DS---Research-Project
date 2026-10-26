# Heart Failure Machine Learning & Clinical Prognostication Research

A multi-phase comparative research repository evaluating machine learning methodologies for heart failure risk factor analysis and survival prediction across both baseline literature and large-scale multimodal clinical cohorts.

---

## Repository Structure

```
.
├── README.md                                  # Repository overview and roadmap
├── .gitignore                                 # Git ignore rules for ML/Jupyter artifacts
├── cardiovascular_hf_dataset-3.csv            # 50,000-patient multimodal dataset (68 clinical dimensions)
│
├── old_dataset_experiments/                   # PHASE 1: 299-Patient Baseline Reproduction & Audits
│   ├── README.md                              # Detailed guide to baseline versions & findings
│   ├── data/
│   │   └── heart_failure_clinical_records_dataset.csv  # Original UCI cohort (299 patients, 13 features)
│   ├── docs/
│   │   └── A machine learning approach...pdf  # Published paper by Ali et al. (Healthcare Analytics, 2023)
│   ├── v1_paper_replication/                  # Exact paper replication (Table 2–8, Fig 2–8)
│   │   ├── heart_failure_paper_implementation.ipynb
│   │   ├── paper_implementation.py
│   │   ├── generate_and_execute_notebook.py
│   │   ├── research.ipynb
│   │   └── results/                           # Replicated figures (Fig 2 through Fig 8)
│   ├── v2_rigorous_reanalysis/                # Leakage-free methodological re-analysis
│   │   ├── heart_failure_rigorous_reanalysis.ipynb
│   │   └── heart_failure_rigorous_reanalysis_results.xlsx
│   └── v3_deep_learning_reanalysis/           # PyTorch Tabular Deep Learning (MLP) exploration
│       └── heart_failure_deep_learning_reanalysis.ipynb
│
└── experiment/                                # PHASE 2: 50,000-Patient Advanced Multimodal Benchmark
    ├── README.md                              # Benchmark summary and model leaderboard
    ├── PAPER_COMPARISON_AND_IMPROVEMENTS.md   # Systematic methodological audit & advancement report
    ├── 01_eda_and_cohort_profiling.ipynb      # Multimodal EDA, missingness audit, Table 1
    ├── 02_model_development_and_benchmarking.ipynb # 8 competing models, DCA, calibration
    ├── 03_risk_factors_xai_and_survival_analysis.ipynb # TreeSHAP, GMM phenotyping, Cox PH
    └── results/                               # 26 publication figures, tables, and reports
```

---

## Research Phases Overview

### Phase 1: Baseline Literature Reproduction & Critical Re-Analysis (`old_dataset_experiments/`)
Investigates the 2023 study by Ali et al. (*Healthcare Analytics*, Volume 3, 100182) on the 299-patient UCI dataset:
- **v1_paper_replication**: Faithful reconstruction of the original workflow (SMOTE + IQR, DT/RF/XGBoost, GMM clustering, Cox PH).
- **v2_rigorous_reanalysis**: Identification of follow-up time leakage (`time` feature), establishment of leakage-free repeated cross-validation, and realistic baseline performance metrics.
- **v3_deep_learning_reanalysis**: Exploration of small-scale neural network architectures (PyTorch MLP) with class-weighted loss.

### Phase 2: Large-Scale Multimodal Benchmark (`experiment/`)
Addresses the core limitations of the 2023 study by transitioning to a real-world, multimodal cohort of **50,000 patients** with 68 clinical indicators:
- **Comprehensive Modalities**: Demographics, hemodynamics, gold-standard cardiac biomarkers (NT-proBNP, Troponin-I, BNP), renal metrics (eGFR, Creatinine), ECG indicators (QTc, QRS, AFib), and echocardiography (EF %, LV wall thickness).
- **Leakage-Free Protocol**: 80/20 holdout split (40,000 train / 10,000 test) with nested fold transformations.
- **Top Models**: CatBoost (AUROC 0.9639, AUPRC 0.9054) and Multimodal Stacking Ensemble (AUROC 0.9632, AUPRC 0.9032).
- **Translational Explainability**: TreeSHAP beeswarm and non-linear biomarker curves, unsupervised GMM patient subphenotyping, Kaplan-Meier curves, and Cox Proportional Hazards survival analysis.

---

## Getting Started

### Prerequisites
- Python 3.9+
- Recommended packages:
  ```bash
  pip install numpy pandas scikit-learn xgboost lightgbm catboost torch lifelines shap matplotlib seaborn openpyxl
  ```

### Running the Experiments
1. **Old Dataset Replication**:
   - Navigate to [`old_dataset_experiments/v1_paper_replication/`](old_dataset_experiments/v1_paper_replication/) to review the direct replication.
   - Run [`paper_implementation.py`](old_dataset_experiments/v1_paper_replication/paper_implementation.py) to regenerate figures.
2. **Advanced 50k Multimodal Benchmark**:
   - Navigate to [`experiment/`](experiment/) and run notebooks `01`, `02`, and `03` sequentially.
