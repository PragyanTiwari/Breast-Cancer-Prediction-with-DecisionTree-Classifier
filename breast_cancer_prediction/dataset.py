import os
import typer
import pickle
import numpy as np
import pandas as pd
from rich import console
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from breast_cancer_prediction.config import RAW_DATA_DIR, INTERIM_DATA_DIR

# Typer Object
app = typer.Typer()

console = console.Console()


@app.command()
def main(
    # ---- CONFIGURATION ----
    input_path: Path = RAW_DATA_DIR / "breast-cancer-dataset.csv",
    output_path: Path = INTERIM_DATA_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    # -------------DATASET PROCESSING-----------------------
):
    logger.info("dataset.py: START")

    # loading raw dataset
    df = pd.read_csv(input_path)
    df.drop("id", axis=1, inplace=True)

    # train-test split
    X = df.drop("diagnosis", axis=1)
    Y = df["diagnosis"]

    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    logger.info("dataset.py: TRAIN-TEST SPLIT DONE.")

    # standardizing the features
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # labeling features with least/no importance i.e. feature_imp <= 0
    clf = DecisionTreeClassifier(criterion="entropy").fit(x_train, y_train)
    feature_imp = {"features": X.columns, "importance": clf.feature_importances_}

    feature_imp_df = pd.DataFrame(feature_imp)
    least_imp_features = np.where(feature_imp_df["importance"] <= 0, True, False)
    x_train_least = x_train[:, np.argwhere(least_imp_features).flatten()]
    x_test_least = x_test[:, np.argwhere(least_imp_features).flatten()]
    x_train_most = x_train[:, np.argwhere(~least_imp_features).flatten()]
    x_test_most = x_test[:, np.argwhere(~least_imp_features).flatten()]

    logger.info("dataset.py: X LEAST/MOST FEATURES CREATED.")

    # saving interim dataset
    training_file = os.path.join(output_path, "training.pkl")
    with open(training_file, "wb") as f:
        pickle.dump([x_train_most, y_train], f)

    testing_file = os.path.join(output_path, "testing.pkl")
    with open(testing_file, "wb") as f:
        pickle.dump([x_test_most, y_test], f)

    least_imp_feature_data_file = os.path.join(output_path, "least_imp_features.pkl")
    with open(least_imp_feature_data_file, "wb") as f:
        pickle.dump([x_train_least, x_test_least], f)

    logger.success("dataset.py: DONE")

    console.print("STAGE-1-SUCCESS ✅", style="bold blue underline on purple3", justify="center")


if __name__ == "__main__":
    app()
