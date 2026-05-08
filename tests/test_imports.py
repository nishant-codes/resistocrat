import os
import pytest

def test_pipeline_import():
    try:
        import scripts.amr_pipeline
        assert True
    except ImportError as e:
        pytest.fail(f"Could not import scripts.amr_pipeline: {e}")

def test_critical_files():
    assert os.path.exists("streamlit_app.py")
    assert os.path.exists("antibiotic_fullgene_summary.csv")
    assert os.path.exists("genotype_matrix.csv")
    assert os.path.exists("models")

def test_models_exist():
    # Check if at least some models are present
    models = [f for f in os.listdir("models") if f.endswith(".joblib")]
    assert len(models) > 0, "No pre-trained models found in models/ directory"
