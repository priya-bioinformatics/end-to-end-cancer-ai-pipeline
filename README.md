# End-to-End Cancer AI Pipeline (TCGA-LIHC)
An end-to-end machine learning and survival analysis pipeline for TCGA-LIHC RNA-seq data, integrating explainable AI and deep learning–based latent feature modeling.
## Overview
This project implements a complete computational pipeline for analyzing liver cancer (LIHC) using RNA-seq gene expression and clinical survival data.
The pipeline covers:
Tumor vs Normal classification
Explainable AI for model interpretability
Survival analysis using gene-level and latent features
Deep learning–based representation learning
## Features
RNA-seq data preprocessing
XGBoost classification with Optuna hyperparameter tuning
Model evaluation using:
ROC-AUC
Calibration curve
Brier score
Explainable AI using SHAP
Gene-based survival analysis:
Cox proportional hazards model
Kaplan–Meier analysis
Autoencoder-based latent space survival modeling
## Dataset
Total samples: 424
Tumor: 371
Normal: 53
Data type: STAR-aligned, log₂-normalized gene-level RNA-seq expression
Cancer type: TCGA-LIHC (Liver Hepatocellular Carcinoma)
## Data Sources
Gene expression data:[TCGA-LIHC STAR-aligned gene-level RNA-seq data](https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LIHC.star_counts.tsv.gz)
Clinical Survival data: [TCGA-LIHC clinical survival data](https://xenabrowser.net/datapages/?dataset=TCGA-LIHC.survival.tsv&host=https%3A%2F%2Fgdc.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)
## How to Run
Run the pipeline sequentially from top to bottom.
