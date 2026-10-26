# Cardiovascular Heart Failure Research Experiment: Advancing Beyond the 2023 Paper

## Overview
This directory contains a complete, methodologically rigorous, and large-scale improvement of the heart failure research paper:
> **Original Paper:** *"A machine learning approach for risk factors analysis and survival prediction of Heart Failure patients"*  
> **Journal:** *Healthcare Analytics*, Volume 3, November 2023, 100182  
> **Original Dataset:** 299 patients, 12 features (`heart_failure_clinical_records_dataset.csv`)

### The New Benchmark
- **Dataset:** 50,000 patients, 68 multimodal clinical dimensions (`cardiovascular_hf_dataset-3.csv`)
- **Clinical Modalities:** Demographics, lifestyle/SES, hemodynamics, gold-standard cardiac biomarkers (NT-proBNP, Troponin-I, BNP), renal function (eGFR, Creatinine), electrophysiology (QTc, QRS, AFib), echocardiography (EF %, LV wall thickness, Cardiomegaly), HF symptoms, and full GDMT medications.
- **Methodology:** 100% leakage-free stratified cross-validation, zero patient discarding, full probabilistic calibration, Decision Curve Analysis (DCA), TreeSHAP explainability, unsupervised GMM phenotyping, and multivariable Cox survival regression.

---

## Notebook Structure
The experiment is organized into three fully executed and self-contained Jupyter Notebooks:

### 1. [`01_eda_and_cohort_profiling.ipynb`](file:///c:/Users/VIVEK/Desktop/DS/Research/experiment/01_eda_and_cohort_profiling.ipynb)
- **Cohort Profiling:** 50,000 patients across Inpatient, Preventive Cardiology, and Annual Checkup settings.
- **Baseline Characteristics (Table 1):** $\chi^2$ and Welch's $t$-test p-values across all 68 clinical indicators for Low-Risk vs High-Risk cohorts.
- **Missing Data Audit:** Characterization of clinical missingness (MAR patterns in biomarkers and echo).
- **Correlation Clustermap:** Multi-organ disease interactions (cardiorenal and cardiometabolic syndrome clusters).
- **Comparison Table:** Direct audit of 299 vs 50,000 patient cohorts.

### 2. [`02_model_development_and_benchmarking.ipynb`](file:///c:/Users/VIVEK/Desktop/DS/Research/experiment/02_model_development_and_benchmarking.ipynb)
- **Leakage-Free Pipeline:** 80/20 train/test holdout split (40,000 train / 10,000 test) with all imputers and scalers fit strictly on training folds.
- **8 Competing Architectures:**
  - Paper Baselines: Decision Tree (DT), Random Forest (RF), Gradient Boosting (GB)
  - Advanced Tree Ensembles: XGBoost, LightGBM, CatBoost
  - Deep Learning: PyTorch Multimodal Residual Tabular Neural Network (MLP)
  - Blending Architecture: Multimodal Stacking Ensemble
- **Multi-Metric Evaluation:** AUROC, AUPRC (PR-AUC), Accuracy, Balanced Accuracy, Recall/Sensitivity, Specificity, Precision, F1-Score, and Brier Score.
- **Probability Calibration:** Reliability diagrams and Expected Calibration Error (ECE).
- **Decision Curve Analysis (DCA):** Net Clinical Benefit evaluation across decision thresholds ($10\% - 50\%$).
- **Ablation Study:** Systematic modality breakdown (Biomarkers vs Vitals vs Demographics).

### 3. [`03_risk_factors_xai_and_survival_analysis.ipynb`](file:///c:/Users/VIVEK/Desktop/DS/Research/experiment/03_risk_factors_xai_and_survival_analysis.ipynb)
- **Explainable AI (SHAP):** TreeSHAP beeswarm summary, global rankings, and non-linear dependence curves for NT-proBNP, Ejection Fraction, Troponin-I, eGFR, and SBP.
- **Unsupervised Clinical Phenotyping:** GMM clustering identifying 4 distinct subphenotypes (Cardiorenal HFrEF, Hypertensive HFpEF, Metabolic Cardiopathy, and Low-Risk Controls) projected onto 2D PCA.
- **Longitudinal Survival Analysis:** Kaplan-Meier survival curves stratified by phenotypes, NT-proBNP tiers, EF classes, and renal function.
- **Multivariable Cox Proportional Hazards Model:** Hazard Ratios (HR) with 95% Confidence Intervals and Forest Plot.

---

## Key Experimental Results Summary

| Model | AUROC | AUPRC | Recall / Sensitivity (%) | Specificity (%) | F1-Score (%) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost** | **0.9639** | **0.9054** | **92.60%** | **86.66%** | **80.39%** | **0.0802** |
| **Stacking Ensemble** | **0.9632** | **0.9032** | **92.18%** | **87.05%** | **80.56%** | **0.0792** |
| **XGBoost** | 0.9628 | 0.9033 | 91.80% | 86.74% | 80.05% | 0.0807 |
| **Gradient Boosting** | 0.9624 | 0.9025 | 80.03% | 93.57% | 80.74% | 0.0690 |
| **LightGBM** | 0.9620 | 0.8999 | 91.07% | 87.55% | 80.47% | 0.0791 |
| **Deep Tabular MLP** | 0.9585 | 0.8946 | 93.45% | 83.44% | 77.75% | 0.0928 |
| **Random Forest** | 0.9553 | 0.8826 | 87.96% | 87.89% | 79.15% | 0.0899 |
| **Decision Tree** | 0.9393 | 0.8141 | 90.46% | 83.62% | 76.38% | 0.1040 |

*All results validated on an untouched, out-of-sample holdout test cohort of 10,000 clinical records.*

---

## Generated Results & Artifacts
All generated plots and tables are located in [`results/`](file:///c:/Users/VIVEK/Desktop/DS/Research/experiment/results):
- **Excel Spreadsheet:** [`model_benchmark_results.xlsx`](file:///c:/Users/VIVEK/Desktop/DS/Research/experiment/results/model_benchmark_results.xlsx)
- **Detailed Scientific Report:** [`RESEARCH_PAPER_IMPROVEMENT_REPORT.md`](file:///c:/Users/VIVEK/Desktop/DS/Research/experiment/results/RESEARCH_PAPER_IMPROVEMENT_REPORT.md)
- **Paper Comparison & Improvements Document:** [`PAPER_COMPARISON_AND_IMPROVEMENTS.md`](file:///c:/Users/VIVEK/Desktop/DS/Research/experiment/PAPER_COMPARISON_AND_IMPROVEMENTS.md)
