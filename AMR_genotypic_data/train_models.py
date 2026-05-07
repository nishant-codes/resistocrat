#!/usr/bin/env python3
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
RANDOM_STATE   = 42
N_ESTIMATORS   = 200
THRESHOLD      = 0.90
GENO_FILE      = 'genotype_matrix.csv'
PHEN_FILE      = 'amr_phenotype_only.csv'
SUMMARY_FILE   = 'antibiotic_fullgene_summary.csv'
MODELS_DIR     = 'models'

# -------------------------------------------------------------------
# 1. Prepare
# -------------------------------------------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)

# Load genotype matrix (0/1) and phenotype table
geno = pd.read_csv(GENO_FILE, index_col='BioSample')
phen = pd.read_csv(PHEN_FILE, sep='\t', index_col='BioSample', low_memory=False)

# Load your summary of test accuracies
summary = pd.read_csv(SUMMARY_FILE)

# Select only those antibiotics with Test_acc ≥ THRESHOLD
reliable = summary.loc[summary['Test_acc'] >= THRESHOLD, 'Antibiotic'].tolist()
print(f"Will train models for: {reliable}")

# -------------------------------------------------------------------
# 2. Train & Save
# -------------------------------------------------------------------
for ab in reliable:
    # 2a. Binary labels
    y0   = phen[ab]
    mask = y0.isin(['resistant','susceptible'])
    y    = y0[mask].map({'susceptible':0, 'resistant':1})
    
    # 2b. Feature matrix aligned to those samples
    X = geno.loc[y.index]

    # 2c. Train a Random Forest on entire dataset
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    clf.fit(X, y)

    # 2d. Serialize
    outpath = os.path.join(MODELS_DIR, f"{ab}_rf.joblib")
    joblib.dump(clf, outpath)
    print(f"Saved model → {outpath}")
