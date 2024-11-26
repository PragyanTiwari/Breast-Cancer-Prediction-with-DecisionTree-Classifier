import typer
import numpy as np
from pathlib import Path
from loguru import logger
from rich import console
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from breast_cancer_prediction.pca import PCA
from breast_cancer_prediction.utils import load_data, save_data
from breast_cancer_prediction.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

console = console.Console()


# PCA function
def perform_pca(n, X, fit_train):
    """
    Transforms data using PCA to create n principal components.
    Args:
        n: int
            Number of components to keep.
        X: np.ndarray of shape (n_samples, n_features)
            Input data.
        fit_train: np.ndarray of shape (n_samples, n_features)
            Training data used to fit the PCA model.
    Returns:
        x_pca: np.ndarray of shape (n_samples, n_components)
            Transformed data with n principal components.
    """
    pca = PCA(n_components=n)
    pca.fit(fit_train)
    x_pca = pca.transform(X)
    return x_pca


@app.command()
def main(
    # ---- CONFIGURATION ----
    input_path: Path = INTERIM_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
):
    # ---- FEATURE ENGINEERING ----

    logger.info("features.py: START")
    x_train, y_train = load_data(input_path, "training.pkl")
    x_test, y_test = load_data(input_path, "testing.pkl")
    x_train_least, x_test_least = load_data(input_path, "least_imp_features.pkl")

    clf = DecisionTreeClassifier(criterion="entropy")
    scores = []
    max_features = x_train_least.shape[1]
    logger.info("features.py: INTERIM DATA LOADED.")

    # running the loop to find optimal value of n_components
    for _ in range(1, max_features + 1):
        x_pca = perform_pca(_, x_train_least, fit_train=x_train_least)
        clf.fit(x_pca, y_train)
        x_pca_test = perform_pca(_, x_test_least, fit_train=x_train_least)
        y_pred = clf.predict(x_pca_test)
        acc_score = accuracy_score(y_test, y_pred)
        scores.append(acc_score)

    max_score = max(scores)
    n_components = scores.index(max_score) + 1

    # transforming the data using optimal n_components
    x_train_pca = perform_pca(n_components, X=x_train_least, fit_train=x_train_least)
    x_test_pca = perform_pca(n_components, X=x_test_least, fit_train=x_train_least)
    logger.info("features.py: PCA COMPLETED.")

    # compiling the pca features to x_train,x_test important features
    n_dim_x_train = np.concatenate((x_train, x_train_pca), axis=1)
    n_dim_x_test = np.concatenate((x_test, x_test_pca), axis=1)

    # saving the data into processed dir.
    save_data(output_path, filename="n_dim_training.pkl", data=[n_dim_x_train, y_train])
    save_data(output_path, filename="n_dim_testing.pkl", data=[n_dim_x_test, y_test])
    logger.info("features.py: DATA SAVED TO PROCESSED DIR.")

    logger.success("features.py: DONE")

    console.print("STAGE-2-SUCCESS ✅", style="bold blue underline on purple3", justify="center")
    # -----------------------------------------


if __name__ == "__main__":
    app()
