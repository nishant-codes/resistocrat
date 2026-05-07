#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load genotype presence/absence (BioSample index)
geno = pd.read_csv('genotype_matrix.csv', index_col='BioSample')

# 2. Load phenotype wide table (tab-delimited, BioSample index)
phen = pd.read_csv('amr_phenotype_only.csv', sep='\t', index_col='BioSample')

# 3. Choose antibiotic to predict
target_ab = 'meropenem'  
if target_ab not in phen.columns:
    raise ValueError(f"{target_ab} not found in phenotype columns")

# 4. Extract labels, drop missing
y = phen[target_ab].dropna()

# 5. Align X to only those BioSamples with labels
X = geno.loc[y.index]

# 6. Encode labels to 0/1
le = LabelEncoder()
y_enc = le.fit_transform(y)  # e.g. ['susceptible','resistant'] → [0,1]

# 7. Split into train/test (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 8. Train & cross-validate
clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

# 9. Fit & predict
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 10. Output metrics
print(f"Antibiotic: {target_ab}")
print("5-fold CV accuracy:", cv_scores.mean().round(3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
