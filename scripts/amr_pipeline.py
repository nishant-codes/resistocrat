#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
MIN_SAMPLES  = 30    # skip antibiotics with fewer than this many labels
N_ESTIMATORS = 200

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
def load_data():
    # genotype_matrix.csv: rows=BioSample, cols=genes (0/1)
    geno = pd.read_csv('genotype_matrix.csv', index_col='BioSample')
    # amr_phenotype_only.csv: rows=BioSample, cols=antibiotics ('resistant','susceptible',...)
    phen = pd.read_csv('amr_phenotype_only.csv', sep='\t', index_col='BioSample', low_memory=False)
    return geno, phen

# -----------------------------------------------------------------------------
# 2. Evaluate full‐gene RF models over antibiotics
# -----------------------------------------------------------------------------
def evaluate_full_models(geno: pd.DataFrame,
                         phen: pd.DataFrame) -> pd.DataFrame:
    results = []
    for ab in phen.columns:
        y0 = phen[ab]
        mask = y0.isin(['resistant','susceptible'])
        if mask.sum() < MIN_SAMPLES:
            continue

        # Binary labels
        y = y0[mask].map({'susceptible':0, 'resistant':1})
        # Feature matrix for these samples
        X = geno.loc[y.index]

        # Stratified train/test split
        Xtr, Xte, ytr, yte = train_test_split(
            X, y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y
        )

        # Random Forest
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

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. Main
# -----------------------------------------------------------------------------
def main():
    geno, phen   = load_data()
    summary      = evaluate_full_models(geno, phen)
    summary.sort_values('Test_acc', ascending=False, inplace=True)
    summary.to_csv('antibiotic_fullgene_summary.csv', index=False)
    print(summary)

if __name__ == '__main__':
    main()
