import typer
from rich import console
from pathlib import Path
from loguru import logger
from sklearn.tree import DecisionTreeClassifier
from breast_cancer_prediction.utils import load_data, save_data
from breast_cancer_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

console = console.Console()


@app.command()
def main(
    # ---- CONFIGURATION ----
    processed_data_path: Path = PROCESSED_DATA_DIR,
    model_path: Path = MODELS_DIR,
):
    # ---- TRAINING ----
    logger.info("train.py: START")

    x_train, y_train = load_data(data_dir=processed_data_path, filename="n_dim_training.pkl")

    clf = DecisionTreeClassifier()

    # NOTE : tuned parameters found using indiviual hyperparameter exploration in notebook_4
    params = {
        "criterion": "entropy",
        "max_depth": 3,
        "min_samples_split": 0.01,
        "min_samples_leaf": 0.01,
        "random_state": 42,
    }

    clf.set_params(**params)
    clf.fit(x_train, y_train)

    logger.info("train.py: MODEL TUNED.")

    save_data(data=clf, data_dir=model_path, filename="model.pkl")
    logger.info("train.py: MODEL SAVED TO MODEL_DIR.")

    logger.success("train.py: DONE")

    console.log(clf, log_locals=False, justify="center", highlight=True)
    console.print("STAGE-3-SUCCESS ✅", style="bold blue underline on purple3", justify="center")
    # -----------------------------------------


if __name__ == "__main__":
    app()
