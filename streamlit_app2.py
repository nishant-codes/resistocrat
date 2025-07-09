#!/usr/bin/env python3
import streamlit as st
import subprocess
import pandas as pd
import joblib
import tempfile
import os
import matplotlib.pyplot as plt

# --- Page setup ---
st.set_page_config(page_title="AMR Predictor", layout="wide")
st.title("Resistocrat – AMR Prediction from Genome Assembly")

# --- Determine reliable antibiotics dynamically ---
summary = pd.read_csv("antibiotic_fullgene_summary.csv")
reliable = summary.loc[summary["Test_acc"] >= 0.90, "Antibiotic"].tolist()
st.sidebar.markdown("## Models included")
st.sidebar.write(reliable)

# --- File uploader ---
uploaded = st.file_uploader("Upload genome assembly (FASTA)", type=["fasta","fa","fna"])
if not uploaded:
    st.info("Please upload a FASTA assembly to get started.")
    st.stop()

# --- Save assembly to temp file ---
with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
    tmp.write(uploaded.read())
    asm_path = tmp.name

# --- Annotate AMR genes with ABRicate ---
st.info("Running ABRicate (CARD)...")
out_tsv = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv").name
subprocess.run(
    ["abricate", "--db", "card", asm_path, "--quiet"],
    stdout=open(out_tsv, "w"),
    stderr=subprocess.DEVNULL,
    check=True
)

# --- Parse the ABRicate output ---
df_hits = pd.read_csv(out_tsv, sep="\t", dtype=str)
genes_found = sorted(df_hits["GENE"].dropna().unique())

# --- Build feature vector ---
geno_cols = pd.read_csv("genotype_matrix.csv", nrows=0).columns.tolist()[1:]
feat = {g: (1 if g in genes_found else 0) for g in geno_cols}
X = pd.DataFrame([feat], index=["sample"])

# --- Predict for each antibiotic ---
preds = {}
for ab in reliable:
    model_file = os.path.join("models", f"{ab}_rf.joblib")
    clf = joblib.load(model_file)
    p = clf.predict(X)[0]
    preds[ab] = "Resistant" if p == 1 else "Susceptible"

result_df = pd.DataFrame.from_dict(preds, orient="index", columns=["Prediction"])
st.subheader("Predicted Phenotypes")
st.table(result_df)

# --- Plot 1: Resistant vs. Susceptible bar chart ---
st.subheader("Prediction Summary")
counts = result_df["Prediction"].value_counts()
fig1, ax1 = plt.subplots()
counts.plot(kind="bar", ax=ax1)
ax1.set_ylabel("Number of antibiotics")
st.pyplot(fig1)

# --- Detected genes list ---
st.subheader("Detected AMR Genes")
st.write(", ".join(genes_found))

# --- Table of detected genes ---
with st.expander("Show detected genes as a table"):
    st.table(pd.DataFrame(genes_found, columns=["Gene"]))
