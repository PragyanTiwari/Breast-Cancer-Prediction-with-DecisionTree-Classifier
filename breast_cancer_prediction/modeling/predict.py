import os
import typer
import pandas as pd
from pathlib import Path
from rich import console
from loguru import logger
from breast_cancer_prediction.utils import load_data
from sklearn.metrics import classification_report, roc_auc_score
from breast_cancer_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULT_DATA_DIR

app = typer.Typer()

console = console.Console()


@app.command()
def main(
    # ---- CONFIGURATION ----
    processed_data_path: Path = PROCESSED_DATA_DIR,
    model_path: Path = MODELS_DIR,
    prediction_path: Path = RESULT_DATA_DIR,
):
    # ---- TRAINING ----
    logger.info("predict.py: START")

    # loading the model & testing dataset
    clf = load_data(data_dir=model_path, filename="model.pkl")
    x_test, y_test = load_data(data_dir=processed_data_path, filename="n_dim_testing.pkl")

    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)[:, 1].round(decimals=3)

    prediction_res = {
        "actual": y_test.to_numpy(),
        "predicted": y_pred,
        "probability": y_pred_proba,
    }

    prediction_df = pd.DataFrame(prediction_res)
    logger.info("predict.py: PREDICTION CREATED.")

    res_path = os.path.join(prediction_path, "prediction.csv")

    prediction_df.to_csv(res_path, index=False)
    logger.info("predict.py: PREDICTION SAVED TO RESULTS FOLDER.")

    logger.success("predict.py: DONE")

    console.print("STAGE-4-SUCCESS ✅", style="bold blue underline on purple3", justify="center")

    console.print("\n[bold red]Classification Report:[/bold red]\n")

    report = classification_report(y_test, y_pred)
    console.print("\n", f"[bold] {report} [/bold]", "\n", justify="center", highlight=True)

    roc_score = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]).round(3)
    console.print(f"roc_auc_score: {roc_score} \n", style="magenta")

    # -----------------------------------------


if __name__ == "__main__":
    app()
