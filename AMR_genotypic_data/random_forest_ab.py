#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RANDOM_STATE = 42
MIN_SAMPLES   = 30     # skip antibiotics with fewer than this many labels
N_ESTIMATORS  = 200

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
geno = pd.read_csv('genotype_matrix.csv', index_col='BioSample')
phen = pd.read_csv('amr_phenotype_only.csv', sep='\t', index_col='BioSample')

# ------------------------------------------------------------
# 2. Loop over antibiotics (full‐gene model only)
# ------------------------------------------------------------
results = []

for ab in phen.columns:
    y_raw = phen[ab]
    mask = y_raw.isin(['resistant','susceptible'])
    if mask.sum() < MIN_SAMPLES:
        continue

    y = y_raw[mask].map({'susceptible':0, 'resistant':1})
    X = geno.loc[y.index]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    cv_acc   = cross_val_score(rf, Xtr, ytr, cv=5, scoring='accuracy').mean()
    rf.fit(Xtr, ytr)
    test_acc = rf.score(Xte, yte)

    results.append({
        'Antibiotic':  ab,
        'N_samples':   int(mask.sum()),
        'CV_acc':      round(cv_acc,3),
        'Test_acc':    round(test_acc,3)
    })

# ------------------------------------------------------------
# 3. Summarize and save
# ------------------------------------------------------------
summary = pd.DataFrame(results).sort_values('Test_acc', ascending=False)
summary.to_csv('antibiotic_fullgene_summary.csv', index=False)
print(summary)
