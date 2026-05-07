# 🧬 Resistocrat: AMR Phenotype Predictor

**Resistocrat** is a machine learning-based tool designed to predict Antimicrobial Resistance (AMR) phenotypes directly from bacterial genome assemblies. By correlating genotypic information (identified AMR genes) with pre-trained Random Forest models, it provides rapid and accurate predictions for a wide range of antibiotics.

---

## 🚀 Key Features

- **Automated Annotation**: Seamless integration with `ABRicate` and the `CARD` (Comprehensive Antibiotic Resistance Database) for precise AMR gene identification.
- **High-Accuracy ML Models**: Utilizes pre-trained Random Forest classifiers optimized for high-reliability predictions (accuracy ≥ 90%).
- **Interactive Dashboard**: A user-friendly Streamlit web interface for easy file uploads and result visualization.
- **Comprehensive Reporting**: Displays predicted phenotypes (Resistant/Susceptible) alongside visual summaries and detailed gene hit tables.

---

## 🛠️ Workflow

1.  **Input**: User uploads a genome assembly file (FASTA format).
2.  **Genotype Extraction**: The system runs `ABRicate` to annotate AMR genes against the CARD database.
3.  **Feature Mapping**: Detected genes are mapped to a high-dimensional genotype vector based on the project's reference matrix.
4.  **Phenotype Prediction**: Optimized Random Forest models process the genotype vector to predict resistance/susceptibility for multiple antibiotics.
5.  **Visualization**: Results are rendered in real-time, providing both a summary bar chart and a detailed breakdown.

---

## 📦 Installation

Since the project relies on `ABRicate` (available via Bioconda), we recommend using **Conda** for environment management.

### 1. Clone the Repository
```bash
git clone https://github.com/nishant-codes/resistocrat.git
cd resistocrat
```

### 2. Set Up Environment
```bash
# Create the environment from the provided YAML file
conda env create -f environment.yml

# Activate the environment
conda activate amr-env
```

### 3. Verify ABRicate
Ensure `abricate` is correctly installed and the CARD database is available:
```bash
abricate --check
abricate --list
```

---

## 💻 Usage

To launch the interactive web application:

```bash
streamlit run streamlit_app2.py
```

1.  Open your browser to the local URL provided by Streamlit (usually `http://localhost:8501`).
2.  **Upload** your genome assembly (e.g., `.fasta`, `.fna`, `.fa`).
3.  Wait for the analysis to complete.
4.  Explore the **Predicted Phenotypes** table and **AMR Gene** summaries.

---

## 🏗️ Training & Research

For researchers looking to retrain models or explore the underlying data, we have included the full training pipeline:

-   **Data Directory (`AMR_genotypic_data/`)**: Contains raw and intermediate data files used for feature engineering.
-   **Training Script (`scripts/amr_pipeline.py`)**: A comprehensive Python script that:
    1.  Loads the genotype and phenotype matrices.
    2.  Filters antibiotics based on sample size.
    3.  Trains Random Forest models with 5-fold cross-validation.
    4.  Generates the `antibiotic_fullgene_summary.csv` accuracy report.

To retrain the models:
```bash
python scripts/amr_pipeline.py
```

---

## 📊 Data & Models

- **Genotype Matrix**: A 90MB+ reference matrix (`genotype_matrix.csv`) defining the gene features used for training.
- **Predictive Models**: Pre-trained Random Forest classifiers located in the `models/` directory.
- **Reliability Filter**: Only models with verified high test accuracy are exposed in the UI.

---

## 📚 Tech Stack

- **Bioinformatics**: [ABRicate](https://github.com/tseemann/abricate), [CARD Database](https://card.mcmaster.ca/)
- **Machine Learning**: Scikit-learn, Joblib
- **Data Science**: Pandas, NumPy, Matplotlib
- **Web Framework**: Streamlit

---

## 📄 License

(c) All rights reserved.

---
*Developed as a tool for rapid AMR screening from genomic data.*
