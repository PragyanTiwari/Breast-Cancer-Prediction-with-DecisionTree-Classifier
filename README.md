# Breast Cancer Prediction with DecisionTree
<img src=".readme-utils\cancer prediction.png" width="750" height="360">

## 🔮 The Project aims to ...

This project focuses on training and fine-tuning a Decision Tree Classifier to predict breast cancer outcomes as either positive or negative based on a diverse range of significant attributes.

 The dataset used for this project is the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), containing features for the prediction of the class : **Malignant(+ve)** or **Benign(-ve)**.


### ** More about the work :**
---

A basic overview of breast cancer dataset is covered in [📒**notebook-1**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/01-data-overview-breast-cancer-classification.ipynb). Simple plots to show distribution of features. 

Built a basic DecisionTree with default parameters and trained on the training dataset in [📒**notebook-2**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/02-decision-tree-model-training.ipynb). 

The least important features found in the previous notebook are then reduced to n optimal dimensions using Principal Component Analysis (**PCA**) in [📒**notebook-3**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/03-pca-feature-engineering.ipynb). The top n principal components having the highest eigenvalues are chosen for model training. A sample is shown below explaining data variance by top 3 eigenvectors. 

<img src="figures\principal_components.png" width="750" height="360" />

 Further along, in [📒**notebook-4**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/04-hyperparameter-tuning.ipynb), hyperparameters are tuned and optimal parameters are then used for the prediction. **RESULT**: *Individual hyper-parameter training show better results than GridSearch CV.* 

**Prediction performance of the tuned model:**
<img src=".readme-utils\report cli.png" width="1000" height="200" alt="Model Performance">



