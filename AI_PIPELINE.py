# NOTE:
# This script is designed as a sequential pipeline.
# Please run from top to bottom without skipping sections.
# Library imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import roc_curve, auc
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import clone
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc
import joblib
# Create output directories if they do not exist
os.makedirs("data", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/genes", exist_ok=True)
os.makedirs("results/explainability", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
# Data Preparation
# Load TCGA-LIHC RNA-seq STAR count data
print("Working directory:", os.getcwd())
expr = pd.read_csv(
    "../Data/TCGA-LIHC.star_counts.tsv",
    sep="\t",
    index_col=0
)
# Display dataset dimensions (genes × samples)
print(expr.shape)
# Transpose matrix so that samples are rows and genes are columns
expr_t = expr.T
# Extract TCGA sample type from barcode
# '01' = Primary Tumor, '11' = Solid Tissue Normal
expr_t["sample_type"] = expr_t.index.str[13:15]
# Create binary classification label
# 1 = tumor, 0 = normal
expr_t["label"] = (expr_t["sample_type"] == "01").astype(int)
# Separate features (gene expression) and target labels
X = expr_t.drop(columns=["sample_type", "label"])
y = expr_t["label"]
# Verify feature matrix dimensions
print(X.shape)
# Check class distribution (tumor vs normal)
y.value_counts()
# Stratified train–test split to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
# Print train and test set sizes
print(X_train.shape, X_test.shape)
# Save splits for reproducibility and downstream analyses
X_train.to_csv("data/X_train.csv")
X_test.to_csv("data/X_test.csv")
y_train.to_csv("data/y_train.csv")
y_test.to_csv("data/y_test.csv")

# XGBoost Classification
# Preprocessing pipeline
preprocess = Pipeline([
    ("var", VarianceThreshold()), 
    ("scaler", StandardScaler())
])
# Optuna function for XGBoost hyperparameter optimization
def objective(trial):

    xgb_params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42
    }
    xgb = XGBClassifier(**xgb_params)
    pipe = Pipeline([
        ("prep", preprocess),
        ("model", xgb)
    ])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        pipe.fit(X_tr, y_tr)
        y_val_prob = pipe.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_val_prob))

    return np.mean(aucs)
# Run Optuna hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
# Retrieve best hyperparameters
best_params = study.best_params
print("Best XGBoost params:", best_params)
# Train final XGBoost model using optimized parameters
xgb_best = XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)
# Final pipeline with preprocessing and optimized model
pipe_xgb = Pipeline([
    ("prep", preprocess),
    ("model", xgb_best)
])
# Fit model on full training dataset
pipe_xgb.fit(X_train, y_train)
# Save trained pipeline for reproducibility
joblib.dump(pipe_xgb, "models/xgb_pipeline.pkl")
# Predict class probabilities and labels
y_prob = pipe_xgb.predict_proba(X_test)[:, 1]
y_pred = pipe_xgb.predict(X_test)
# ROC-AUC score
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
# Classification metrics
print(classification_report(y_test, y_pred))
# Save classification report
pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True)
).to_csv("results/metrics/xgb_classification_report.csv")
# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
# Save ROC-AUC value
pd.DataFrame({"roc_auc": [roc_auc]}).to_csv(
    "results/metrics/xgb_roc_auc.csv",
    index=False
)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost – Tumor vs Normal")
plt.legend()
plt.savefig("figures/roc_xgboost_tumor_vs_normal.png", dpi=300)
plt.show()
# Calibration (reliability) curve
prob_true, prob_pred = calibration_curve(
    y_test,
    y_prob,
    n_bins=10,
    strategy="uniform"
)
plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, marker="o", label="XGBoost")
plt.plot([0,1], [0,1], "--", label="Perfect calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Reliability Curve (Model Calibration)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/xgb_reliability_curve.png", dpi=300)
plt.show()
# Brier score for probabilistic accuracy
brier = brier_score_loss(y_test, y_prob)
print("Brier score:", brier)
# Save Brier score
pd.DataFrame({"brier_score": [brier]}).to_csv(
    "results/metrics/xgb_brier_score.csv",
    index=False
)
# Explainability & gene importance
# Use a small subset of training samples to reduce SHAP computation cost
X_shap = X_train.sample(
    min(50, X_train.shape[0]),
    random_state=42
)
# Apply trained preprocessing pipeline (variance filter + scaling)
X_shap_prep = pipe_xgb.named_steps["prep"].transform(X_shap)
# Convert back to DataFrame with gene names
X_shap_prep = pd.DataFrame(
    X_shap_prep,
    columns=preprocess.get_feature_names_out(),
    index=X_shap.index
)
# Compute SHAP values for XGBoost model
explainer = shap.TreeExplainer(pipe_xgb.named_steps["model"])
shap_values = explainer.shap_values(X_shap_prep)
# Visualize top contributing genes
shap.summary_plot(
    shap_values,
    X_shap_prep,
    max_display=20
)
# Save SHAP outputs
np.save("results/explainability/shap_values_xgb.npy", shap_values)
X_shap_prep.to_csv("results/explainability/shap_input_samples.csv")
# Retrieve genes retained after variance filtering
var_mask = pipe_xgb.named_steps["prep"].named_steps["var"].get_support()
selected_genes = X.columns[var_mask]
# Extract and rank XGBoost feature importances
importances = pipe_xgb.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({
    "Gene": selected_genes,
    "Importance": importances
}).sort_values("Importance", ascending=False)
# Save ranked gene list
imp_df.to_csv(
    "results/genes/xgb_gene_importance.csv",
    index=False
)
# Save top 20 genes
top_genes = imp_df.head(20)["Gene"].tolist()
pd.Series(top_genes).to_csv(
    "results/genes/top20_genes_xgb.txt",
    index=False,
    header=False
)
top_genes
# Survival Analysis
# Load TCGA-LIHC clinical survival data
clinical = pd.read_csv(
    "../Data/TCGA-LIHC.survival.tsv",
    sep="\t"
)
clinical.head()
# Standardize column names
clinical = clinical.rename(columns={
    "sample": "sample_id",
    "OS.time": "OS_time",
    "OS": "OS_event"
})
# Select top genes for Cox regression
top_genes = imp_df.head(8)["Gene"].tolist()
# Restrict analysis to tumor samples
X_tumor = X[y == 1]
# Merge gene expression with survival data
surv_df = pd.merge(
    X_tumor[top_genes].reset_index().rename(columns={"index": "sample_id"}),
    clinical,
    on="sample_id",
    how="inner"
)
print("Survival DF shape:", surv_df.shape)
surv_df.head()
# Prepare dataset for Cox proportional hazards model
cox_df = surv_df.drop(columns=["sample_id"])
# Ensure numeric types and remove invalid values
cox_df[top_genes] = cox_df[top_genes].apply(pd.to_numeric, errors="coerce")
cox_df["OS_time"] = pd.to_numeric(cox_df["OS_time"], errors="coerce")
cox_df["OS_event"] = pd.to_numeric(cox_df["OS_event"], errors="coerce")
# Remove missing or non-positive survival times
cox_df = cox_df.dropna(subset=["OS_time", "OS_event"])
cox_df = cox_df[cox_df["OS_time"] > 0]
print("Final Cox DF shape:", cox_df.shape)
cox_df["OS_event"].value_counts()
# Drop non-feature identifier if present
cox_df = cox_df.drop(columns=["_PATIENT"], errors="ignore")
# Standardize gene expression before Cox regression
gene_cols = [c for c in cox_df.columns if c not in ["OS_time", "OS_event"]]
scaler = StandardScaler()
cox_df[gene_cols] = scaler.fit_transform(cox_df[gene_cols])
# Fit penalized Cox proportional hazards model
cph = CoxPHFitter(penalizer=5.0)
cph.fit(
    cox_df,
    duration_col="OS_time",
    event_col="OS_event"
)
cph.print_summary()
# Save Cox regression results
cph.summary.to_csv("results/metrics/cox_gene_summary.csv")
# Compute patient risk scores from Cox model
cox_df["risk_score"] = cph.predict_partial_hazard(cox_df)
# Define high- and low-risk groups using median cutoff
median_risk = cox_df["risk_score"].median()
cox_df["risk_group"] = np.where(
    cox_df["risk_score"] > median_risk,
    "High Risk", "Low Risk"
)
# Plot Kaplan–Meier survival curves
kmf = KaplanMeierFitter()
plt.figure(figsize=(7,5))
for group in ["High Risk", "Low Risk"]:
    mask = cox_df["risk_group"] == group
    kmf.fit(
        cox_df.loc[mask, "OS_time"],
        cox_df.loc[mask, "OS_event"],
        label=group
    )	 
    kmf.plot_survival_function()
plt.title("Kaplan–Meier Survival Curve (AI-Derived Risk)")
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.tight_layout()
plt.show()
# Log-rank test to compare survival distributions
high = cox_df[cox_df["risk_group"] == "High Risk"]
low = cox_df[cox_df["risk_group"] == "Low Risk"]
result = logrank_test(
    high["OS_time"], low["OS_time"],
    event_observed_A=high["OS_event"],
    event_observed_B=low["OS_event"]
)
print("Log-rank p-value:", result.p_value)
# Concordance index to assess survival prediction accuracy
c_index = concordance_index(
    cox_df["OS_time"],
    -cox_df["risk_score"],
    cox_df["OS_event"]
)
print("C-index:", c_index)
cph.summary[["coef", "exp(coef)", "p"]].sort_values("p").head(10)
# Time-dependent AUC for survival model
# Create survival object
y_surv = Surv.from_dataframe(
    event="OS_event",
    time="OS_time",
    data=cox_df
)
risk_scores = cox_df["risk_score"].values
# Evaluate AUC at 1-, 3-, and 5-year time points
time_points = np.array([365, 1095, 1825])
auc_times, auc_values = cumulative_dynamic_auc(
    y_surv,
    y_surv,
    risk_scores,
    time_points
)
# Handle scalar edge case
if np.isscalar(auc_values):
    auc_times = np.array([auc_times])
    auc_values = np.array([auc_values])
# Print time-dependent AUC values
for t, auc in zip(auc_times, auc_values):
    t_val = float(np.atleast_1d(t)[0])
    auc_val = float(np.atleast_1d(auc)[0])
    print(f"Time-dependent AUC at time={t_val:.3f}: {auc_val:.3f}")
# Plot dynamic AUC
plt.figure(figsize=(6,4))
plt.plot(
    np.atleast_1d(auc_times),
    np.atleast_1d(auc_values),
    marker="o"
)
plt.xlabel("Time (days)")
plt.ylabel("Time-dependent AUC")
plt.title("Dynamic AUC of Cox Survival Model")
plt.ylim(0.5, 1.0)
plt.grid(True)
plt.tight_layout()
plt.show()
# Autoencoder (deep learning)
# Clone preprocessing pipeline for deep learning
preprocess_dl = clone(preprocess)
# Preprocess training data (variance filtering + scaling)
X_train_prep = preprocess_dl.fit_transform(X_train)
# Convert preprocessed data to PyTorch tensor
X_train_tensor = torch.tensor(X_train_prep, dtype=torch.float32)
# Create dataset and data loader
dataset = TensorDataset(X_train_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Define autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
# Initialize autoencoder
model = Autoencoder(X_train_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
# Train autoencoder using reconstruction loss
for epoch in range(40):
    for (batch,) in loader:
        recon, _ = model(batch)
        loss = loss_fn(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
# Save trained autoencoder weights
torch.save(
    model.state_dict(),
    "models/autoencoder_state_dict.pt"
)
# Extract latent features from training data
with torch.no_grad():
    _, Z = model(X_train_tensor)
# Save latent representations
np.save("results/latent_features.npy", Z.numpy())
# Survival analysis (latent space)
# Apply preprocessing to tumor samples before encoding
X_tumor_prep = preprocess.transform(X_tumor)
# Generate latent representations using trained autoencoder
with torch.no_grad():
    _, Z = model(torch.tensor(X_tumor_prep, dtype=torch.float32))
Z = Z.numpy()
print("Latent feature shape:", Z.shape)
# Create DataFrame of latent features
latent_cols = [f"Z{i}" for i in range(Z.shape[1])]
latent_df = pd.DataFrame(
    Z,
    index=X_tumor.index,   # sample IDs
    columns=latent_cols
)
latent_df.head()
# Merge latent features with clinical survival data
cox_latent = pd.merge(
    latent_df.reset_index().rename(columns={"index": "sample_id"}),
    clinical,
    on="sample_id",
    how="inner"
)
# Remove invalid survival records
cox_latent = cox_latent.dropna(subset=["OS_time", "OS_event"])
cox_latent = cox_latent[cox_latent["OS_time"] > 0]
print("Latent Cox DF shape:", cox_latent.shape)
# Retain only latent features and survival columns
latent_cols = [c for c in cox_latent.columns if c.startswith("Z")]
cox_latent_clean = cox_latent[
    latent_cols + ["OS_time", "OS_event"]
].copy()
cox_latent_clean.dtypes
# Fit penalized Cox model on latent features
cph_latent = CoxPHFitter(penalizer=1.0)
cph_latent.fit(
    cox_latent_clean,
    duration_col="OS_time",
    event_col="OS_event"
)
cph_latent.print_summary()
# Save latent Cox model summary
cph_latent.summary.to_csv(
    "results/metrics/cox_latent_summary.csv"
)
# Compute risk scores and stratify patients
cox_latent_clean["risk_score"] = cph_latent.predict_partial_hazard(
    cox_latent_clean[latent_cols]
)
cox_latent_clean["risk_group"] = np.where(
    cox_latent_clean["risk_score"] > cox_latent_clean["risk_score"].median(),
    "High Risk",
    "Low Risk"
)
# Kaplan–Meier survival curves based on latent risk groups
kmf = KaplanMeierFitter()
plt.figure(figsize=(7,5))
for g in ["High Risk", "Low Risk"]:
    m = cox_latent_clean["risk_group"] == g
    kmf.fit(
        cox_latent_clean.loc[m, "OS_time"],
        cox_latent_clean.loc[m, "OS_event"],
        label=g
    )
    kmf.plot_survival_function()
plt.title("Autoencoder-based Survival Stratification")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.tight_layout()
plt.show()
