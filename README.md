# end-to-end-cancer-ai-pipeline
End-to-end machine learning and survival analysis pipeline for TCGA-LIHC RNA-seq data with explainable AI and deep learning.
## Overview
This project implements a complete computational pipeline for:
- Tumor vs Normal classification
- Explainable AI
- Survival analysis
- Deep learning-based latent feature modeling
The analysis is performed on TCGA-LIHC RNA-seq gene expression and clinical survival data.
## Features
- RNA-seq data preprocessing
- XGBoost classification with Optuna hyperparameter tuning
- Model evaluation using ROC-AUC, calibration curve, and Brier score
- Explainable AI using SHAP
- Gene-based survival analysis (Cox proportional hazards, Kaplan–Meier)
- Autoencoder-based latent space survival modeling
## Dataset
424 samples (371 tumor, 53 normal)
STAR-aligned, log₂-normalized gene-level expression
## Data Sources
Gene expression data:[TCGA-LIHC STAR-aligned gene-level RNA-seq data](https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LIHC.star_counts.tsv.gz)
Clinical Survival data: [TCGA-LIHC clinical survival data](https://xenabrowser.net/datapages/?dataset=TCGA-LIHC.survival.tsv&host=https%3A%2F%2Fgdc.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)


---

## How to Run

Run the pipeline sequentially from top to bottom:
