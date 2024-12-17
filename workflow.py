import subprocess

def stages():
    """Running sequence of python scripts to build prediction."""
    scripts = [
        "breast_cancer_prediction.config",
        "breast_cancer_prediction.dataset",
        "breast_cancer_prediction.features",
        "breast_cancer_prediction.modeling.train",
        "breast_cancer_prediction.modeling.predict"
    ]

    for script in scripts:
        subprocess.run(["python", "-m", script])