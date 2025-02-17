# 🎗️ Breast Cancer Prediction with DecisionTree
<img src=".readme-utils\cancer prediction.png" width="750" height="360">

## 🔮 The Project aims to ...

This project focuses on training and fine-tuning a Decision Tree Classifier to predict breast cancer outcomes as either positive or negative based on a diverse range of significant attributes.

<img src=".readme-utils\infographic.png" width="700" height="600">

 The dataset used for this project is the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), containing features for the prediction of the class : **Malignant(+ve)** or **Benign(-ve)**.


### **More about the work 📔:**
---

A basic overview of breast cancer dataset is covered in [📒**notebook-1**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/01-data-overview-breast-cancer-classification.ipynb). Simple plots to show distribution of features. 

Built a basic DecisionTree with default parameters and trained on the training dataset in [📒**notebook-2**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/02-decision-tree-model-training.ipynb). 

The least important features found in the previous notebook are then reduced to n optimal dimensions using Principal Component Analysis (**PCA**) in [📒**notebook-3**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/03-pca-feature-engineering.ipynb). The top n principal components having the highest eigenvalues are chosen for model training. A sample is shown below explaining data variance by top 3 eigenvectors. 

<img src="figures\principal_components.png" width="750" height="360" />

 Further along, in [📒**notebook-4**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/04-hyperparameter-tuning.ipynb), hyperparameters are tuned and optimal parameters are then used for the prediction. **RESULT**: *Individual hyper-parameter training show better results than GridSearch CV.* 

**Prediction performance of the ✨tuned model:**
<img src=".readme-utils\report cli.png" width="1000" height="200" alt="Model Performance">

To understand how the tuned model works and how it is making predictions, in [📒**notebook-5**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/05-tree-model-explainability_SHAP.ipynb), SHAP library is used for Model Interpretability. Global and 
The SHAP library is used to achieve Model Interpretability, enabling both global and local analyses of the optimized model's behavior. The **Decision Plot** below illustrates how individual features contribute to the prediction process, providing a clear understanding of the model's decision-making logic.

<img src="figures\shap_decision_plot.png" width="700" height="550" />


## 📝 Installation Guide (Building Predictions)

- **Clone the repository**
```shell
git clone https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier.git
```

- **Using Makefile :**

```shell
# install uv if not
pip install --upgrade uv
```

```shell
# to create virtual env
make create_environment
```

```shell
# install python dependencies
make requirements
```
```shell
 # build predictions
make breast_cancer_prediction
```

- **Using [uv](https://docs.astral.sh/uv/) (If not Makefile):**

```shell
# to create virtual env
uv venv
```

```shell
# install python dependencies
uv add --requirements 'requirements.txt' --dev
```
```shell
 # build predictions
uv run make_predictions
```
<img src=".readme-utils\terminal.png" width="1200" height="300">

❕The output will be saved as `predictions.csv` in data\result dir.





       
