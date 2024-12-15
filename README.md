# Breast Cancer Prediction with DecisionTree
<img src=".readme-utils\cancer prediction.png" width="750" height="360">

## 🔮 The Project aims to ...

This project focuses on training and fine-tuning a Decision Tree Classifier to predict breast cancer outcomes as either positive or negative based on a diverse range of significant attributes.

 The dataset used for this project is the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), containing features for the prediction of the class : **Malignant(+ve)** or **Benign(-ve)**.


### **About the** [**notebooks**](notebooks):
---

[📒**notebook-1**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/01-data-overview-breast-cancer-classification.ipynb) covers basic overview of the breast cancer dataset. 

Built a basic DecisionTree with default parameters and trained on the training dataset in [📒**notebook-2**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/02-decision-tree-model-training.ipynb). 

The least important features found in the previous notebook are then reduced to 3 dimensions using Principal Component Analysis (**PCA**) in [📒**notebook-3**](https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier/blob/master/notebooks/03-pca-feature-engineering.ipynb) . The top 3 principal components having the highest eigenvalues are chosen and their variance ratio explained by each vector is show below:

<img src=".figures\principal_components.png" width="750" height="360" />




