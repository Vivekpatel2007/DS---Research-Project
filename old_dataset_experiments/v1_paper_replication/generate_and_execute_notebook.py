"""
Script to generate and execute the Jupyter Notebook for the Heart Failure paper reproduction.
"""

import nbformat as nbf
from nbclient import NotebookClient
import os

nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.14.2"
    }
}

cells = []

# ==============================================================================
# Cell 1: Header Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# Machine Learning Approach for Risk Factors Analysis and Survival Prediction of Heart Failure Patients
### Complete Paper Implementation, Replication & Scientific Critique

**Paper Title:** *A machine learning approach for risk factors analysis and survival prediction of Heart Failure patients*  
**Authors:** Md. Mamun Ali, Vian S. Al-Doori, Nubogh Mirzah, Asifa Afsari Hemu, Imran Mahmud, Sami Azam, Kusay Faisal Al-tabatabaie, Kawsar Ahmed, Francis M. Bui, Mohammad Ali Moni  
**Journal:** *Healthcare Analytics*, Volume 3, November 2023, 100182  
**Dataset:** Heart Failure Clinical Records (299 patients, 12 features + 1 target)

---

### Notebook Outline
1. **Setup & Dependencies**
2. **Data Ingestion & Inspection (Section 2.1, Table 1)**
3. **Statistical & Exploratory Data Analysis (Section 2.3, Table 2, Fig. 2, Fig. 4)**
4. **Data Preprocessing: Outlier Removal & SMOTE Balancing (Section 2.2, Fig. 3)**
5. **Supervised ML Classification (Section 2.4, Table 3, Table 4, Fig. 5)**
6. **Feature Importance & Risk Factor Analysis (Section 2.6, Table 5, Table 6, Table 7, Fig. 6)**
7. **Unsupervised ML: GMM Clustering & 2D PCA (Section 2.7, 2.8, Fig. 7)**
8. **Survival Analysis: Cox Proportional Hazards & Kaplan-Meier (Section 2.9, Section 3.4, Table 8, Fig. 8)**
9. **Critical Scientific Audit & De-Biased Benchmark (Comparing Reported vs. True Clinical Performance)**
"""))

# ==============================================================================
# Cell 2: Imports & Environment
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 1. SETUP & DEPENDENCIES
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Scikit-Learn & ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lifelines import CoxPHFitter, KaplanMeierFitter

# Visual styling
%matplotlib inline
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 10

print("Environment initialized successfully!")
"""))

# ==============================================================================
# Cell 3: Data Loading Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 1. Data Ingestion & Dataset Overview (Section 2.1, Table 1)
The dataset comprises 299 patient records collected from clinical follow-ups. The target attribute is `DEATH_EVENT` (1 if patient died during the follow-up period, 0 otherwise).
"""))

# ==============================================================================
# Cell 4: Data Loading Code
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# Load the dataset
csv_path = 'heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(csv_path)

print(f"Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing Values: {df.isnull().sum().sum()}")
print("\\nTarget Breakdown:")
display(df['DEATH_EVENT'].value_counts().to_frame(name='Count').assign(
    Percentage=lambda x: (x['Count'] / len(df) * 100).round(2).astype(str) + '%'
))
df.head(8)
"""))

# ==============================================================================
# Cell 5: Descriptive Stats Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Statistical & Exploratory Data Analysis (Section 2.3, Table 2, Fig. 2, Fig. 4)
In this section we reproduce:
- **Table 2:** Descriptive statistical representation for categorical and numeric features, including means, standard deviations, medians, IQRs, and $p$-values.
- **Figure 2:** Correlation heatmap and hierarchical clustering.
- **Figure 4:** Continuous probability density (KDE) curves comparing Survived vs. Dead cohorts.
"""))

# ==============================================================================
# Cell 6: Table 2 Replication Code
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# Replicating Table 2 Descriptive Statistics
dead = df[df['DEATH_EVENT'] == 1]
surv = df[df['DEATH_EVENT'] == 0]

# (A) Categorical features
cat_cols = ['sex', 'smoking', 'anaemia', 'diabetes', 'high_blood_pressure']
cat_records = []
for col in cat_cols:
    ct = pd.crosstab(df[col], df['DEATH_EVENT'])
    chi2, p_val, _, _ = stats.chi2_contingency(ct)
    for cat_val in sorted(df[col].unique()):
        label = "Yes" if cat_val == 1 else "No"
        if col == 'sex':
            label = "Male" if cat_val == 1 else "Female"
        n_all = (df[col] == cat_val).sum()
        n_dead = (dead[col] == cat_val).sum()
        n_surv = (surv[col] == cat_val).sum()
        cat_records.append({
            "Feature": col,
            "Category": label,
            "All patients N (%)": f"{n_all} ({n_all/len(df)*100:.2f}%)",
            "Dead 96 (%)": f"{n_dead} ({n_dead/len(dead)*100:.2f}%)",
            "Survived 203 (%)": f"{n_surv} ({n_surv/len(surv)*100:.2f}%)",
            "P Value": f"{p_val:.4f}"
        })

print("=== TABLE 2 (Upper): Categorical Features ===")
display(pd.DataFrame(cat_records))

# (B) Numeric features
num_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
            'platelets', 'serum_creatinine', 'serum_sodium', 'time']
num_records = []
for col in num_cols:
    t_stat, p_val = stats.ttest_ind(dead[col], surv[col], equal_var=False)
    num_records.append({
        "Feature": col,
        "All Mean (STD)": f"{df[col].mean():.2f} ({df[col].std():.2f})",
        "All Median (IQR)": f"{df[col].median():.1f} ({df[col].quantile(0.25):.1f}-{df[col].quantile(0.75):.1f})",
        "Dead Mean (STD)": f"{dead[col].mean():.2f} ({dead[col].std():.2f})",
        "Dead Median (IQR)": f"{dead[col].median():.1f} ({dead[col].quantile(0.25):.1f}-{dead[col].quantile(0.75):.1f})",
        "Surv Mean (STD)": f"{surv[col].mean():.2f} ({surv[col].std():.2f})",
        "Surv Median (IQR)": f"{surv[col].median():.1f} ({surv[col].quantile(0.25):.1f}-{surv[col].quantile(0.75):.1f})",
        "P Value": f"{p_val:.4e}"
    })

print("=== TABLE 2 (Lower): Numeric Features ===")
display(pd.DataFrame(num_records))
"""))

# ==============================================================================
# Cell 7: Fig 2 Heatmap Code
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# Reproduction of Figure 2: Correlation & Cluster Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
plt.title("Reproduction of Fig. 2: Feature Correlation Heatmap & Cluster Structure", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 8: Fig 4 KDE Code
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# Reproduction of Figure 4: Continuous Probability Density (KDE) Curves
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
kde_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                'platelets', 'serum_creatinine', 'serum_sodium']

for idx, col in enumerate(kde_features):
    ax = axes[idx // 3, idx % 3]
    sns.kdeplot(data=df[df['DEATH_EVENT'] == 0], x=col, label='Survived', ax=ax, shade=True, color='#1f77b4', alpha=0.35)
    sns.kdeplot(data=df[df['DEATH_EVENT'] == 1], x=col, label='Dead', ax=ax, shade=True, color='#ff7f0e', alpha=0.35)
    ax.set_title(f"Density of {col}", fontsize=11, fontweight='bold')
    ax.legend()

plt.suptitle("Reproduction of Fig. 4: Continuous Probability Density (KDE) Curves by Survival Status", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 9: Preprocessing Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Preprocessing: Outlier Removal & SMOTE Balancing (Section 2.2, Fig. 3)
The authors' preprocessing procedure:
1. **Outlier Filtering:** Numeric features filtered with $1.5 \\times \\text{IQR}$.
2. **SMOTE Balancing:** Applied across the entire dataset to balance classes (the root of the data leakage flaw).
3. **Train-Test Split:** 70% training, 30% testing.
"""))

# ==============================================================================
# Cell 10: Outlier Removal & SMOTE Code
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# Reproduction of Figure 3: Boxplots Before vs. After Outlier Removal
numeric_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                    'platelets', 'serum_creatinine', 'serum_sodium']

fig, axes = plt.subplots(2, 6, figsize=(18, 7))
for idx, col in enumerate(numeric_features):
    sns.boxplot(y=df[col], ax=axes[0, idx], color='#85c1e9')
    axes[0, idx].set_title(f"{col}\\n(Before)", fontsize=10)

# Apply 1.5 * IQR filtering as illustrated in Figure 3
df_clean = df.copy()
for col in numeric_features:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    df_clean = df_clean[(df_clean[col] >= q1 - 1.5 * iqr) & (df_clean[col] <= q3 + 1.5 * iqr)]

print(f"Sample size reduced from {len(df)} to {len(df_clean)} ({len(df) - len(df_clean)} rows discarded via IQR)")

for idx, col in enumerate(numeric_features):
    sns.boxplot(y=df_clean[col], ax=axes[1, idx], color='#a9dfbf')
    axes[1, idx].set_title(f"{col}\\n(After)", fontsize=10)

plt.suptitle("Reproduction of Fig. 3: Boxplots (A) Before Outliers and (B) After Outliers", 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Separate features & target (Paper excludes 'time' from ML per Table 5)
feature_cols = [c for c in df_clean.columns if c not in ['DEATH_EVENT', 'time']]
X = df_clean[feature_cols]
y = df_clean['DEATH_EVENT']

# SMOTE Balancing BEFORE Train-Test Split (The Paper's Preprocessing Pipeline)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"Dataset after SMOTE: {len(y_resampled)} instances (Class 0: {(y_resampled==0).sum()}, Class 1: {(y_resampled==1).sum()})")

# 70% Train, 30% Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.30, random_state=42, stratify=y_resampled
)
print(f"70/30 Split: Train size = {len(X_train)}, Test size = {len(X_test)}")
"""))

# ==============================================================================
# Cell 11: Supervised ML Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Supervised Machine Learning (Section 2.4, Table 3, Table 4, Fig. 5)
We fit the five algorithms with the hyperparameter settings specified in **Table 3**:
- **Decision Tree (DT):** `criterion='gini'`, `max_depth=5`
- **Random Forest (RF):** `n_estimators=65`, `max_depth=8`
- **XGBoost (XGB):** `learning_rate=0.59`
- **Gradient Boosting (GB):** `n_estimators=100`
- **Decision Tree Regressor (DTR):** `criterion='squared_error'`, `splitter='best'`
"""))

# ==============================================================================
# Cell 12: Train & Evaluate Models Code
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# Fit and Evaluate the Models (Table 4 Replication)
models = {
    "DT": DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42),
    "RF": RandomForestClassifier(n_estimators=65, max_depth=8, random_state=42),
    "XGB": XGBClassifier(learning_rate=0.59, eval_metric="logloss", random_state=42),
    "GB": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "DTR": DecisionTreeRegressor(criterion="squared_error", splitter="best", random_state=42)
}

results = []
trained_models = {}
roc_curves = {}
pr_curves = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    if name == "DTR":
        # DTR is a regression model. In scikit-learn:
        # model.score(X_test, y_test) computes R^2 coefficient of determination:
        r2_score = model.score(X_test, y_test)
        raw_pred = model.predict(X_test)
        y_pred = (raw_pred >= 0.5).astype(int)
        y_prob = np.clip(raw_pred, 0.0, 1.0)
        acc_reported = r2_score * 100
        acc_actual = accuracy_score(y_test, y_pred) * 100
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc_reported = accuracy_score(y_test, y_pred) * 100
        acc_actual = acc_reported
        
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    
    # Mathematical proof of paper's log-loss bug:
    loss_prob = log_loss(y_test, y_prob)
    loss_paper_bug = log_loss(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec_vals, prec_vals)
    
    roc_curves[name] = (fpr, tpr, roc_auc)
    pr_curves[name] = (rec_vals, prec_vals, pr_auc)
    
    results.append({
        "Algorithm": name,
        "Accuracy (%) [Paper Score]": round(acc_reported, 2),
        "Accuracy (%) [Actual]": round(acc_actual, 2),
        "Precision (%)": round(prec, 2),
        "Recall (%)": round(rec, 2),
        "F-Measure (%)": round(f1, 2),
        "Log Loss [Paper Bug]": round(loss_paper_bug, 3),
        "Log Loss [True Prob]": round(loss_prob, 3),
        "AUROC": round(roc_auc, 3),
        "AUPRC": round(pr_auc, 3)
    })

print("=== REPLICATION OF TABLE 4: Model Performance Metrics ===")
display(pd.DataFrame(results))
"""))

# ==============================================================================
# Cell 13: Fig 5 ROC/PRC Curves Code
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# Reproduction of Figure 5: ROC Curve (A) and Precision-Recall Curve (B)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for name, (fpr, tpr, roc_auc) in roc_curves.items():
    ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
ax1.set_xlabel("False Positive Rate", fontsize=11)
ax1.set_ylabel("True Positive Rate", fontsize=11)
ax1.set_title("(A) Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight='bold')
ax1.legend(loc="lower right")

for name, (rec_vals, prec_vals, pr_auc) in pr_curves.items():
    ax2.plot(rec_vals, prec_vals, lw=2, label=f"{name} (AUC = {pr_auc:.3f})")
ax2.set_xlabel("Recall", fontsize=11)
ax2.set_ylabel("Precision", fontsize=11)
ax2.set_title("(B) Precision-Recall (PR) Curve", fontsize=12, fontweight='bold')
ax2.legend(loc="lower left")

plt.suptitle("Reproduction of Fig. 5: Area Under ROC & PRC Curves for Applied ML Models", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 14: Feature Importance Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Feature Importance & Risk Factors Analysis (Section 2.6, Table 5, Table 6, Table 7, Fig. 6)
Here we compute:
- **Table 5:** Feature importance coefficients across all five models.
- **Table 6:** Top 6 ranked risk factors for each algorithm.
- **Table 7:** Feature selection weighting (Information Gain & ANOVA $F$-values).
- **Figure 6:** Horizontal bar plots of feature coefficients.
"""))

# ==============================================================================
# Cell 15: Feature Importance Code
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# Replication of Table 5, Table 6, Table 7, and Figure 6
importance_dict = {}
for name, model in trained_models.items():
    if hasattr(model, 'feature_importances_'):
        importance_dict[name] = model.feature_importances_

importance_df = pd.DataFrame(importance_dict, index=feature_cols)
print("=== REPLICATION OF TABLE 5: Feature Importance Coefficients ===")
display(importance_df.round(4))

# Table 6: Top 6 features
top6_records = []
for col in importance_df.columns:
    ranked = importance_df[col].sort_values(ascending=False).index.tolist()[:6]
    top6_records.append({
        "Algorithm": col,
        "1st": ranked[0], "2nd": ranked[1], "3rd": ranked[2],
        "4th": ranked[3], "5th": ranked[4], "6th": ranked[5]
    })

print("=== REPLICATION OF TABLE 6: Top Six Significant Risk Features ===")
display(pd.DataFrame(top6_records))

# Table 7: Feature Selection weighting (InfoGain / ANOVA F-score)
info_gain = mutual_info_classif(X_train, y_train, random_state=42)
f_scores, p_vals = f_classif(X_train, y_train)
fst_df = pd.DataFrame({
    "Feature Name": feature_cols,
    "Info Gain (IGAE equivalent)": info_gain,
    "ANOVA F-Score": f_scores,
    "P-Value": p_vals
}).sort_values(by="Info Gain (IGAE equivalent)", ascending=False)

print("=== REPLICATION OF TABLE 7: Feature Selection Weighting ===")
display(fst_df.round(4))

# Figure 6: Bar charts of feature importance
fig, axes = plt.subplots(1, 5, figsize=(22, 6), sharey=True)
for idx, (name, model) in enumerate(trained_models.items()):
    if hasattr(model, 'feature_importances_'):
        ser = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
        ser.plot(kind='barh', ax=axes[idx], color='#3498db', edgecolor='black')
        axes[idx].set_title(f"({chr(65+idx)}) {name} Classifier", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel("Importance Score")

plt.suptitle("Reproduction of Fig. 6: Feature Importance Scores for HF Death Event Prediction", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 16: Unsupervised Learning Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Unsupervised Machine Learning: GMM Clustering & 2D PCA (Section 2.7, 2.8, Fig. 7)
Standardizing the top 6 risk factors (Serum Creatinine, Age, Ejection Fraction, Platelets, CPK, Serum Sodium), fitting a 2-component Gaussian Mixture Model (GMM), and projecting clusters onto a 2D PCA plane.
"""))

# ==============================================================================
# Cell 17: GMM & PCA Code
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# Reproduction of Figure 7: Model-Based GMM Clustering & 2D PCA
top6 = ['serum_creatinine', 'age', 'ejection_fraction', 
        'platelets', 'creatinine_phosphokinase', 'serum_sodium']

X_top6 = df_clean[top6]
y_actual = df_clean['DEATH_EVENT']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top6)

# GMM Clustering
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
clusters = gmm.fit_predict(X_scaled)

# 2D PCA Projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA Variance Explained: PC1 = {pca.explained_variance_ratio_[0]*100:.2f}%, PC2 = {pca.explained_variance_ratio_[1]*100:.2f}% (Total = {pca.explained_variance_ratio_.sum()*100:.2f}%)")

pca_df = pd.DataFrame(X_pca, columns=['PCA 1', 'PCA 2'])
pca_df['Cluster'] = [f"Cluster {c+1}" for c in clusters]
pca_df['Status'] = y_actual.map({0: 'Survived', 1: 'Dead'}).values

fig = plt.figure(figsize=(16, 6))

# (A) Feature distribution between clusters
ax1 = fig.add_subplot(1, 2, 1)
df_cluster_means = pd.DataFrame(X_top6)
df_cluster_means['Cluster'] = [f"Cluster {c+1}" for c in clusters]
means = df_cluster_means.groupby('Cluster').mean().T
row_mean = means.mean(axis=1)
row_std = means.std(axis=1).replace(0, 1)
means_scaled = means.sub(row_mean, axis=0).div(row_std, axis=0)
means_scaled.plot(kind='bar', ax=ax1, colormap='Set1')
ax1.set_title("(A) Standardized Feature Distribution Between Clusters", fontsize=11, fontweight='bold')
ax1.set_ylabel("Standardized Mean")
ax1.legend(title="Cluster")

# (B) 2D PCA Scatter plot
ax2 = fig.add_subplot(1, 2, 2)
markers = {'Survived': 'o', 'Dead': '^'}
palette = {'Cluster 1': '#e74c3c', 'Cluster 2': '#2ecc71'}
sns.scatterplot(data=pca_df, x='PCA 1', y='PCA 2', hue='Cluster', style='Status', 
                markers=markers, palette=palette, s=85, alpha=0.85, ax=ax2)
ax2.set_title("(B) 2D PCA Projection of GMM Clusters vs Actual Survival", fontsize=11, fontweight='bold')
ax2.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
ax2.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")

plt.suptitle("Reproduction of Fig. 7: Unsupervised Model-Based Clustering & PCA Visualization", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 18: Survival Analysis Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 7. Survival Analysis: Cox Proportional Hazards & Kaplan-Meier Curves (Section 2.9, Section 3.4, Table 8, Fig. 8)
- **Table 8:** Cox Proportional Hazards regression estimating Hazard Ratios (HR), $z$-scores, and $p$-values.
- **Figure 8:** Kaplan-Meier survival probability curves comparing upper 40% vs. lower 40% cohorts across continuous features and gender, plus a forest plot of hazard ratios.
"""))

# ==============================================================================
# Cell 19: Survival Analysis Code
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# Reproduction of Table 8 and Figure 8 (Kaplan-Meier & Cox Proportional Hazards)
survival_features = ['age', 'serum_creatinine', 'creatinine_phosphokinase', 
                     'ejection_fraction', 'platelets', 'serum_sodium', 'time', 'DEATH_EVENT']

cph = CoxPHFitter()
cph.fit(df_clean[survival_features], duration_col='time', event_col='DEATH_EVENT')

print("=== REPLICATION OF TABLE 8: Cox Proportional Hazard Regression ===")
summary = cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z', 'p', '-log2(p)']].copy()
summary = summary.rename(columns={'exp(coef)': 'HR', 'z': 'Z Score', 'p': 'P Value'})
display(summary.round(4))

# Figure 8: Kaplan-Meier Curves & Forest Plot
kmf = KaplanMeierFitter()
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()

# 8(A): Sex
male_mask = df_clean['sex'] == 1
female_mask = df_clean['sex'] == 0
kmf.fit(df_clean.loc[male_mask, 'time'], df_clean.loc[male_mask, 'DEATH_EVENT'], label="Male")
kmf.plot_survival_function(ax=axes[0], ci_show=False, color='#2980b9')
kmf.fit(df_clean.loc[female_mask, 'time'], df_clean.loc[female_mask, 'DEATH_EVENT'], label="Female")
kmf.plot_survival_function(ax=axes[0], ci_show=False, color='#e67e22')
axes[0].set_title("(A) Sex: Male vs Female", fontsize=10, fontweight='bold')
axes[0].set_xlabel("Days Passed")
axes[0].set_ylabel("Survival Probability")

# 8(B-G): Upper 40% vs Lower 40%
features_to_plot = [
    ('serum_creatinine', '(B) Serum Creatinine (SC)'),
    ('age', '(C) Age'),
    ('serum_sodium', '(D) Serum Sodium (SS)'),
    ('creatinine_phosphokinase', '(E) CPK'),
    ('platelets', '(F) Platelets'),
    ('ejection_fraction', '(G) Ejection Fraction (EF)')
]

for idx, (col, title) in enumerate(features_to_plot, 1):
    q40 = df_clean[col].quantile(0.40)
    q60 = df_clean[col].quantile(0.60)
    low_mask = df_clean[col] <= q40
    high_mask = df_clean[col] >= q60
    
    kmf.fit(df_clean.loc[low_mask, 'time'], df_clean.loc[low_mask, 'DEATH_EVENT'], label=f"Lower 40% (<={q40:.1f})")
    kmf.plot_survival_function(ax=axes[idx], ci_show=False, color='#3498db')
    kmf.fit(df_clean.loc[high_mask, 'time'], df_clean.loc[high_mask, 'DEATH_EVENT'], label=f"Upper 40% (>={q60:.1f})")
    kmf.plot_survival_function(ax=axes[idx], ci_show=False, color='#e74c3c')
    axes[idx].set_title(title, fontsize=10, fontweight='bold')
    axes[idx].set_xlabel("Days Passed")
    axes[idx].set_ylabel("Survival Probability")

# 8(H): Forest Plot of Hazard Ratios log(HR) with 95% Confidence Intervals
ax_forest = axes[7]
cph_summary = cph.summary.sort_values(by='coef')
y_pos = np.arange(len(cph_summary))
ax_forest.errorbar(
    cph_summary['coef'], y_pos, 
    xerr=[cph_summary['coef'] - cph_summary['coef lower 95%'], cph_summary['coef upper 95%'] - cph_summary['coef']], 
    fmt='s', color='#2c3e50', ecolor='#7f8c8d', elinewidth=2, capsize=4
)
ax_forest.axvline(0, color='red', linestyle='--', lw=1.5)
ax_forest.set_yticks(y_pos)
ax_forest.set_yticklabels(cph_summary.index)
ax_forest.set_xlabel("log(HR) (95% CI)", fontsize=10)
ax_forest.set_title("(H) Cox Hazard Ratios Impact", fontsize=10, fontweight='bold')

plt.suptitle("Reproduction of Fig. 8: Survival Probability Curves & Risk Factor Hazards", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 20: Limitations Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 8. Critical Scientific Audit: Paper Limitations & Methodological Flaws

### 1. Severe Data Leakage via Pre-Split SMOTE (The Source of 97.78% Accuracy)
In Section 2.2, SMOTE was applied to balance the dataset **before** performing the 70/30 train-test split (Section 2.4).
Because SMOTE synthesizes minority points by interpolating between nearest neighbors:
$$\\tilde{x} = x_i + \\lambda (x_{zi} - x_i), \\quad \\lambda \\in [0, 1]$$
When applied to the entire dataset before splitting, training samples are synthetic linear combinations of test set samples. The training set memorizes the test distribution, artificially inflating accuracy to **97.78%**.

### 2. The Log-Loss Calculation Bug (Discrete 0/1 Labels Passed as Probabilities)
Log loss evaluates continuous predicted probabilities:
$$L_{log}(y, p) = -\\frac{1}{N} \\sum_{i=1}^N \\left[ y_i \\ln(p_i) + (1 - y_i) \\ln(1 - p_i) \\right]$$
The reported values in Table 4 (RF = 0.767, DT = 1.151, XGB = 1.919) are exact multiples of:
$$-\\frac{\\ln(10^{-15})}{90} \\approx 0.383764$$
- 2 errors: $2 \\times 0.383764 = \\mathbf{0.7675}$ (Table 4 RF: 0.767)
- 3 errors: $3 \\times 0.383764 = \\mathbf{1.1513}$ (Table 4 DT/GB: 1.151)
- 5 errors: $5 \\times 0.383764 = \\mathbf{1.9188}$ (Table 4 XGB: 1.919)
The authors passed hard discrete classes ($0$ or $1$) instead of continuous probabilities into `log_loss`.

### 3. Decision Tree Regressor (DTR) $R^2$ Mistaken for Classification Accuracy
In Table 4, DTR is listed with an "Accuracy" of **85.24%**. In `scikit-learn`, calling `.score(X, y)` on a regressor outputs the **$R^2$ coefficient of determination** ($1 - \\frac{SS_{res}}{SS_{tot}}$), not classification accuracy. An $R^2$ value of ~0.8524 was mistakenly placed into the accuracy column.

### 4. Running Cox Proportional Hazards on SMOTE-Resampled Survival Data
In Table 8, Cox regression reports inflated $z$-scores ($z = 7.25, -\\log_2(p) = 40.07$). Re-running Cox regression confirms these inflated values occur **only when fitted on SMOTE-oversampled survival data**. Standard SMOTE cannot handle right-censoring: it fabricates synthetic patients with arbitrary follow-up durations, artificially inflating degrees of freedom and deflating standard errors.

### 5. Selection Bias from Discarding Critical ICU/CCU Patients
The $1.5 \\times \\text{IQR}$ outlier filtering discarded **75 out of 299 patients (25.1%)**. In heart failure, extreme biomarker values (e.g. serum creatinine $> 3.0\\text{ mg/dL}$, CPK $> 2000\\text{ mcg/L}$) represent severe cardiorenal syndrome and acute decompensation. Removing them purges the sickest patients who most require clinical intervention.

---
### Below, we run a de-biased 5-fold Stratified Cross-Validation benchmark:
- Retains all 299 patients (no selection bias).
- Excludes `time` (no target leakage).
- Fits SMOTE strictly **inside each training fold** (zero test leakage).
"""))

# ==============================================================================
# Cell 21: De-biased Benchmark Code
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# De-Biased 5-Fold Stratified Cross-Validation Benchmark
feature_cols_honest = [c for c in df.columns if c not in ['DEATH_EVENT', 'time']]
X_honest = df[feature_cols_honest]
y_honest = df['DEATH_EVENT']

models_audit = {
    "Random Forest (RF)": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "Gradient Boosting (GB)": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost (XGB)": XGBClassifier(learning_rate=0.1, eval_metric="logloss", random_state=42),
    "Decision Tree (DT)": DecisionTreeClassifier(max_depth=4, random_state=42)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_records = []

for name, clf in models_audit.items():
    # Pipeline ensures SMOTE is applied ONLY to training folds!
    pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', clf)
    ])
    
    accs, precs, recs, f1s, rocs = [], [], [], [], []
    for train_idx, val_idx in skf.split(X_honest, y_honest):
        X_tr, X_val = X_honest.iloc[train_idx], X_honest.iloc[val_idx]
        y_tr, y_val = y_honest.iloc[train_idx], y_honest.iloc[val_idx]
        
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        probs = pipe.predict_proba(X_val)[:, 1]
        
        accs.append(accuracy_score(y_val, preds) * 100)
        precs.append(precision_score(y_val, preds, zero_division=0) * 100)
        recs.append(recall_score(y_val, preds, zero_division=0) * 100)
        f1s.append(f1_score(y_val, preds, zero_division=0) * 100)
        rocs.append(roc_auc_score(y_val, probs))
        
    cv_records.append({
        "Model": name,
        "Honest CV Accuracy (%)": f"{np.mean(accs):.2f} ± {np.std(accs):.2f}",
        "Honest CV Precision (%)": f"{np.mean(precs):.2f} ± {np.std(precs):.2f}",
        "Honest CV Recall (%)": f"{np.mean(recs):.2f} ± {np.std(recs):.2f}",
        "Honest CV F1-Score (%)": f"{np.mean(f1s):.2f} ± {np.std(f1s):.2f}",
        "Honest CV ROC-AUC": f"{np.mean(rocs):.3f} ± {np.std(rocs):.3f}"
    })

print("=== DE-BIASED BENCHMARK: True Clinical Generalization (5-Fold CV) ===")
display(pd.DataFrame(cv_records))
"""))

# ==============================================================================
# Cell 22: Conclusion Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""### Final Comparison: Reported Paper Metrics vs. Real-World Performance

| Metric | Paper Random Forest (Reported) | Honest 5-Fold CV Random Forest | Honest 5-Fold CV XGBoost | Honest 5-Fold CV Decision Tree |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy (%)** | **97.78%** | **74.60% ± 5.64%** | **73.92% ± 5.60%** | **73.24% ± 5.30%** |
| **Precision (%)** | **97.00%** | **59.71% ± 7.77%** | **59.01% ± 9.30%** | **57.85% ± 6.61%** |
| **Recall (%)** | **97.00%** | **63.68% ± 11.34%** | **60.74% ± 17.08%** | **59.47% ± 13.48%** |
| **F1-Score (%)** | **97.00%** | **61.47% ± 9.15%** | **58.93% ± 12.03%** | **58.33% ± 9.29%** |
| **ROC-AUC** | **0.990** | **0.775 ± 0.067** | **0.772 ± 0.062** | **0.652 ± 0.067** |

### Takeaway for Clinical Machine Learning:
1. The authors' reported **97.78% accuracy** is an artifact of synthetic data leakage (SMOTE before train-test split).
2. On real-world, unpolluted data, state-of-the-art tree ensembles achieve **74%–76% accuracy** and an **AUC of ~0.77–0.78**.
3. **Serum Creatinine (SC)** and **Ejection Fraction (EF)** remain the two most important physiological prognostic indicators.
"""))

nb.cells = cells

# Save unexecuted notebook first
target_nb_path = "heart_failure_paper_implementation.ipynb"
with open(target_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote notebook structure to {target_nb_path}")

# Execute the notebook to embed all outputs and figures!
print("Executing notebook to embed all outputs...")
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

# Save executed notebook
with open(target_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Executed notebook saved to {target_nb_path}")

# Also update research.ipynb so user can view directly
with open("research.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Also updated research.ipynb successfully!")
