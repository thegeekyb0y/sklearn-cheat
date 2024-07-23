![image](https://github.com/user-attachments/assets/eca48e41-93eb-4c4c-bc8a-af669cee2967)


Welcome to the **Scikit-learn Cheatsheet** repository! This repository serves as a quick reference guide for various machine learning models and techniques available in the Scikit-learn library.

![Python Version](https://img.shields.io/badge/python-3.7%2B-brightgreen.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## Contribution 💡
- If you have a suggestion that would make this repository better or want to add more resources or any links are not working, please fork the repo and create a [create a pull request.](https://github.com/thegeekyb0y/sklearn-cheat/edit/main/README.md) 
- You can also simply open an [issue](https://github.com/thegeekyb0y/sklearn-cheat/issues/new)
- Read the Contribution guidelines [here](https://github.com/thegeekyb0y/sklearn-cheat/CONTRIBUTIONS.md)
  
# Table of Contents

## Data Preprocessing
- [Handling Missing Values](#handling-missing-values)
  - [SimpleImputer](#simpleimputer)
  - [IterativeImputer](#iterativeimputer)
- [Feature Scaling](#feature-scaling)
  - [StandardScaler](#standardscaler)
  - [MinMaxScaler](#minmaxscaler)
- [Encoding Categorical Features](#encoding-categorical-features)
  - [OneHotEncoder](#onehotencoder)
  - [LabelEncoder](#labelencoder)
- [Feature Engineering](#feature-engineering)
  - [PolynomialFeatures](#polynomialfeatures)
  - [FunctionTransformer](#functiontransformer)

## Supervised Learning
### Classification
- [LogisticRegression](#logisticregression)
- [LinearSVC](#linearsvc)
- [KNeighborsClassifier](#kneighborsclassifier)
- [SVC](#svc)
- [DecisionTreeClassifier](#decisiontreeclassifier)
- [RandomForestClassifier](#randomforestclassifier)
- [GaussianNB](#gaussiannb)
- [Discriminant Analysis](#discriminant-analysis)

### Regression
- [LinearRegression](#linearregression)
- [Ridge](#ridge)
- [Lasso](#lasso)
- [ElasticNet](#elasticnet)
- [KNeighborsRegressor](#kneighborsregressor)
- [SVR](#svr)
- [DecisionTreeRegressor](#decisiontreeregressor)
- [RandomForestRegressor](#randomforestregressor)

## Unsupervised Learning
### Clustering
- [KMeans](#kmeans)
- [AgglomerativeClustering](#agglomerativeclustering)
- [DBSCAN](#dbscan)
- [GaussianMixture](#gaussianmixture)

### Dimensionality Reduction
- [PCA](#pca)
- [TSNE](#tsne)
- [UMAP](#umap)

## Deep Learning
### Neural Networks
- [Multilayer Perceptron (MLP)](#multilayer-perceptron-mlp)
- [Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)
- [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
- [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
- [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
- [Transformers](#transformers)

### Loss Functions
- [Mean Squared Error (MSE)](#mean-squared-error-mse)
- [Categorical Cross-Entropy](#categorical-cross-entropy)
- [Binary Cross-Entropy](#binary-cross-entropy)

### Optimization Algorithms
- [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
- [Adam](#adam)
- [RMSProp](#rmsprop)

## Model Selection and Evaluation
- [Train-Test Split](#train-test-split)
- [Cross-Validation](#cross-validation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
  - [GridSearchCV](#gridsearchcv)
  - [RandomizedSearchCV](#randomizedsearchcv)

## Utility Functions
- [Metrics](#metrics)
  - [accuracy_score](#accuracy-score)
  - [f1_score](#f1-score)
  - [precision_score](#precision-score)
  - [recall_score](#recall-score)
  - [roc_auc_score](#roc-auc-score)
  - [confusion_matrix](#confusion-matrix)

---
 
## Connect with Me 🤝
- <a href="https://www.bento.me/adityatiwari" target="_blank">All my socials</a>

--- 

## Data Preprocessing

### Handling Missing Values
- **SimpleImputer**: Fills in missing values using specified strategies such as mean, median, or most frequent.
  
```python
  from sklearn.impute import SimpleImputer
  import numpy as np

  data = np.array([[1, 2], [np.nan, 3], [7, 6]])
  imputer = SimpleImputer(strategy='mean')
  imputed_data = imputer.fit_transform(data)
```
- **Iterative Imputer**: Unlike SimpleImputer, which uses a single value to fill in missing data, IterativeImputer models each feature with missing values as a function of other features in the dataset. It iteratively predicts missing values based on other available data, which can lead to more accurate imputations, especially in datasets with complex relationships among features.
  
```python
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer
  import numpy as np

  data = np.array([[1, 2], [np.nan, 3], [7, 6]])
  imputer = IterativeImputer()
  imputed_data = imputer.fit_transform(data)
```
--- 
## Feature Scaling
- **StandardScaler**: This scaler standardizes features by removing the mean and scaling to unit variance. It is particularly useful when the features have different units or scales, as it ensures that each feature contributes equally to the distance calculations used in many machine learning algorithms. Standardization is often a prerequisite for algorithms that assume normally distributed data.
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

data = np.array([[1, 2], [2, 3], [3, 4]])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```
