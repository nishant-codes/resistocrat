#!/usr/bin/env python3
import streamlit as st
import subprocess
import pandas as pd
import joblib
import tempfile
import os
import matplotlib.pyplot as plt
from typing import List, Dict

# --- Constants & Config ---
MODELS_DIR = "models"
SUMMARY_FILE = "antibiotic_fullgene_summary.csv"
GENOTYPE_FILE = "genotype_matrix.csv"
FEATURES_CACHE = "features.txt"

st.set_page_config(page_title="Resistocrat - AMR Predictor", layout="wide", page_icon="🧬")
st.title("🧬 Resistocrat – AMR Prediction")

# --- Helper Functions ---

@st.cache_data
def load_config():
    """Load reliability summary and feature columns."""
    if not os.path.exists(SUMMARY_FILE):
        st.error(f"Critical error: {SUMMARY_FILE} not found.")
        st.stop()
    
    summary = pd.read_csv(SUMMARY_FILE)
    reliable = summary.loc[summary["Test_acc"] >= 0.90, "Antibiotic"].tolist()
    
    # Efficiently get feature columns
    if os.path.exists(FEATURES_CACHE):
        with open(FEATURES_CACHE, 'r') as f:
            geno_cols = [line.strip() for line in f if line.strip()]
    else:
        # Fallback to reading CSV header
        geno_cols = pd.read_csv(GENOTYPE_FILE, nrows=0).columns.tolist()[1:]
        # Optionally save to cache for next time
        with open(FEATURES_CACHE, 'w') as f:
            for col in geno_cols:
                f.write(f"{col}\n")
                
    return reliable, geno_cols

@st.cache_resource
def load_model(antibiotic: str):
    """Cache models in memory to avoid repeated disk I/O."""
    model_path = os.path.join(MODELS_DIR, f"{antibiotic}_rf.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def check_dependencies():
    """Verify that ABRicate is installed."""
    try:
        subprocess.run(["abricate", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# --- Main App Logic ---

reliable_abs, feature_names = load_config()

st.sidebar.markdown("## 🛡️ Reliable Models")
st.sidebar.info(f"Loaded {len(reliable_abs)} high-accuracy models (Test Acc ≥ 90%).")
with st.sidebar.expander("Show Antibiotics"):
    st.write(reliable_abs)

if not check_dependencies():
    st.error("⚠️ **ABRicate not found!** Please ensure ABRicate is installed and available in your PATH.")
    st.info("You can install it via conda: `conda install -c bioconda abricate`")
    st.stop()

uploaded = st.file_uploader("Upload genome assembly (FASTA)", type=["fasta", "fa", "fna"])

if uploaded:
    # Use a temporary directory to manage multiple temp files
    with tempfile.TemporaryDirectory() as tmp_dir:
        asm_path = os.path.join(tmp_dir, "input.fasta")
        out_tsv = os.path.join(tmp_dir, "results.tsv")
        
        with open(asm_path, "wb") as f:
            f.write(uploaded.getbuffer())
        
        with st.status("🔍 Analyzing Genome...", expanded=True) as status:
            st.write("Running ABRicate (CARD database)...")
            try:
                subprocess.run(
                    ["abricate", "--db", "card", asm_path, "--quiet"],
                    stdout=open(out_tsv, "w"),
                    stderr=subprocess.PIPE,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                st.error(f"ABRicate failed: {e.stderr.decode()}")
                st.stop()
            
            st.write("Parsing AMR gene hits...")
            df_hits = pd.read_csv(out_tsv, sep="\t", dtype=str)
            genes_found = sorted(df_hits["GENE"].dropna().unique())
            
            if not genes_found:
                st.warning("No AMR genes detected by ABRicate.")
            else:
                st.success(f"Detected {len(genes_found)} AMR genes.")

            st.write("Generating phenotype predictions...")
            # Build feature vector
            feat = {g: (1 if g in genes_found else 0) for g in feature_names}
            X = pd.DataFrame([feat], index=["sample"])
            
            preds = {}
            progress_bar = st.progress(0)
            for idx, ab in enumerate(reliable_abs):
                clf = load_model(ab)
                if clf:
                    p = clf.predict(X)[0]
                    preds[ab] = "Resistant" if p == 1 else "Susceptible"
                progress_bar.progress((idx + 1) / len(reliable_abs))
            
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # --- Display Results ---
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Predicted Phenotypes")
            result_df = pd.DataFrame.from_dict(preds, orient="index", columns=["Prediction"])
            st.dataframe(result_df, height=400, use_container_width=True)

        with col2:
            st.subheader("📊 Prediction Summary")
            counts = result_df["Prediction"].value_counts()
            fig, ax = plt.subplots()
            counts.plot(kind="bar", ax=ax, color=['#ff4b4b', '#00cc96'][:len(counts)])
            ax.set_ylabel("Number of antibiotics")
            st.pyplot(fig)

        # --- Gene Details ---
        st.divider()
        st.subheader("🧬 Detected AMR Genes")
        if genes_found:
            st.info(", ".join(genes_found))
            with st.expander("View Hit Details"):
                st.table(pd.DataFrame(genes_found, columns=["Gene Symbol"]))
        else:
            st.write("No genes detected.")
else:
    st.info("Please upload a bacterial genome assembly in FASTA format to begin analysis.")

st.markdown("---")
st.caption("© 2024 Resistocrat Team | Data source: CARD Database | Engine: Random Forest")
