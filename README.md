![image](https://github.com/user-attachments/assets/eca48e41-93eb-4c4c-bc8a-af669cee2967)


Welcome to the **Scikit-learn Cheatsheet** repository! This repository serves as a quick reference guide for various machine learning models and techniques available in the Scikit-learn library.

![Python Version](https://img.shields.io/badge/python-3.7%2B-brightgreen.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=thegeekyb0y.sklearn-cheat)

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

# Data Preprocessing

## Handling Missing Values
### SimpleImputer
Fills in missing values using specified strategies such as mean, median, or most frequent.
  
```python
  from sklearn.impute import SimpleImputer
  import numpy as np

  data = np.array([[1, 2], [np.nan, 3], [7, 6]])
  imputer = SimpleImputer(strategy='mean')
  imputed_data = imputer.fit_transform(data)
```
### Iterative Imputer 
Unlike SimpleImputer, which uses a single value to fill in missing data, IterativeImputer models each feature with missing values as a function of other features in the dataset. It iteratively predicts missing values based on other available data, which can lead to more accurate imputations, especially in datasets with complex relationships among features.
  
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
### StandardScaler: 
This scaler standardizes features by removing the mean and scaling to unit variance. It is particularly useful when the features have different units or scales, as it ensures that each feature contributes equally to the distance calculations used in many machine learning algorithms. Standardization is often a prerequisite for algorithms that assume normally distributed data.
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

data = np.array([[1, 2], [2, 3], [3, 4]])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```


### MinMaxScaler: 
MinMaxScaler transforms features by scaling them to a specified range, typically [0, 1]. This is particularly useful when the distribution of the data is not Gaussian and when you want to preserve the relationships between the data points. It is commonly used in neural networks, where inputs are often expected to be in a specific range.
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([[1, 2], [2, 3], [3, 4]])
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
```
--- 

## Encoding Categorical Features
### OneHotEncoder: 
This encoder is used to convert categorical variables into a format that can be provided to machine learning algorithms, which typically require numerical input. OneHotEncoder creates binary columns for each category, allowing the model to treat them as separate features. This is particularly useful for non-ordinal categorical data, where the categories do not have a natural order.
```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

data = np.array([['red'], ['green'], ['blue']])
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(data).toarray()
```
### LabelEncoder: 
LabelEncoder is used to convert categorical labels into numeric form. It assigns a unique integer to each category, which is useful for ordinal data where the order matters. This transformation is often required when preparing categorical data for machine learning algorithms that cannot handle non-numeric input.
```python
from sklearn.preprocessing import LabelEncoder
import numpy as np

data = np.array(['red', 'green', 'blue'])
encoder = LabelEncoder()
encoded_data = encoder.fit_transform(data)
```
---

## Feature Engineering
### PolynomialFeatures: 
This transformer generates polynomial and interaction features. It is particularly useful for capturing non-linear relationships in the data, especially in regression tasks. By adding polynomial features, you can enhance the capability of linear models to fit more complex patterns.
```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

data = np.array([[1, 2], [3, 4]])
poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(data)
```

### FunctionTransformer: 
FunctionTransformer allows you to apply any custom function to your dataset. This is useful for preprocessing steps that are not covered by existing transformers. You can define your own transformation logic and integrate it seamlessly into your machine learning pipeline.
```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

data = np.array([[1, 2], [3, 4]])
transformer = FunctionTransformer(np.log1p)
transformed_data = transformer.fit_transform(data)
```

---

# Supervised Learning
## Classification
### Logistic Regression:
A statistical model that uses a logistic function to model binary dependent variables. It is effective for binary classification problems and is widely used due to its simplicity and interpretability.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
```

### LinearSVC
A linear Support Vector Classifier optimized for large datasets. It is particularly useful for high-dimensional data and works well when the classes are linearly separable.
``` python
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = LinearSVC()
model.fit(X_train, y_train)
```

### KNeighborsClassifier
A non-parametric method used for classification. It classifies data points based on the majority class among its k-nearest neighbors. It is effective for small datasets and when the decision boundary is irregular.
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
```


### SVC
Support Vector Classifier that finds the optimal hyperplane that maximizes the margin between different classes. It works well for both linear and non-linear data.
```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
```


### DecisionTreeClassifier
A model that uses a tree-like graph of decisions to classify data. It is easy to interpret and visualize, making it suitable for both classification and regression tasks.

``` python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

### RandomForestClassifier
An ensemble method that builds multiple decision trees and merges them together to improve accuracy and control overfitting. It is robust and effective for a variety of classification tasks.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
```



### GaussianNB
A Naive Bayes classifier based on applying Bayes' theorem with strong (naive) independence assumptions. It is particularly effective for large datasets and works well with normally distributed features.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = GaussianNB()
model.fit(X_train, y_train)
```


## Regression

### LinearRegression
A fundamental statistical technique used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship and is effective for predicting continuous outcomes.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
```
### Ridge
Ridge regression is a type of linear regression that includes a regularization term to prevent overfitting. It is particularly useful when dealing with multicollinearity among features.
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```
### Lasso
Lasso regression is similar to ridge regression but uses L1 regularization, which can shrink some coefficients to zero, effectively performing variable selection. It is useful when you want a simpler model.
```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
```
### ElasticNet
ElasticNet combines the penalties of both Lasso and Ridge regression. It is useful when there are multiple features correlated with each other, providing a balance between the two regularization techniques.
```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)
```


### KNeighborsRegressor
KNeighborsRegressor is a non-parametric regression method that predicts the target value based on the average of the k-nearest neighbors. It is effective for capturing non-linear relationships.
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)
```


### SVR
Support Vector Regression (SVR) is an extension of SVC that applies the principles of SVM to regression problems. It is effective for high-dimensional data and can model non-linear relationships using kernel functions.
```python
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = SVR(kernel='linear')
model.fit(X_train, y_train)
```


### DecisionTreeRegressor
DecisionTreeRegressor is a model that uses a tree structure to predict continuous outcomes. It is easy to interpret and can capture non-linear relationships in the data.
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
```


### RandomForestRegressor
RandomForestRegressor is an ensemble method that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. It is robust and effective for various regression tasks.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)
```
--- 

# Unsupervised Learning

## Clustering

### KMeans
KMeans is a popular clustering algorithm that partitions data into k distinct clusters based on feature similarity. It is widely used for exploratory data analysis and is effective for large datasets.

```python
from sklearn.cluster import KMeans
import numpy as np

data = np.array([[1, 2], [1, 4], [1, 0],
                 [4, 2], [4, 0], [4, 4]])
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
clusters = kmeans.labels_
```


### AgglomerativeClustering
Agglomerative Clustering is a hierarchical clustering method that builds a tree of clusters by iteratively merging the closest pairs of clusters. It is useful for discovering nested clusters in the data.

``` python
from sklearn.cluster import AgglomerativeClustering
import numpy as np

data = np.array([[1, 2], [1, 4], [1, 0],
                 [4, 2], [4, 0], [4, 4]])
model = AgglomerativeClustering(n_clusters=2)
clusters = model.fit_predict(data)
```

### DBSCAN
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together points that are closely packed together while marking as outliers points that lie alone in low-density regions. It is effective for identifying clusters of varying shapes and sizes.

```python
from sklearn.cluster import DBSCAN
import numpy as np

data = np.array([[1, 2], [1, 4], [1, 0],
                 [4, 2], [4, 0], [4, 4]])
model = DBSCAN(eps=1, min_samples=2)
clusters = model.fit_predict(data)
```

### GaussianMixture
Gaussian Mixture Models (GMM) are probabilistic models that assume all data points are generated from a mixture of several Gaussian distributions with unknown parameters. GMM is useful for soft clustering where each point can belong to multiple clusters.

```python
from sklearn.mixture import GaussianMixture
import numpy as np

data = np.array([[1, 2], [1, 4], [1, 0],
                 [4, 2], [4, 0], [4, 4]])
gmm = GaussianMixture(n_components=2)
gmm.fit(data)
clusters = gmm.predict(data)
```


## Dimensionality Reduction

### PCA (Principal Component Analysis)
PCA is a technique used to reduce the dimensionality of a dataset while preserving as much variance as possible. It is commonly used for data visualization and preprocessing before applying machine learning algorithms.

```python
from sklearn.decomposition import PCA
import numpy as np

data = np.array([[1, 2], [2, 3], [3, 4]])
pca = PCA(n_components=1)
reduced_data = pca.fit_transform(data)
```
### t-SNE (t-distributed Stochastic Neighbor Embedding)
t-SNE is a technique for dimensionality reduction that is particularly well-suited for visualizing high-dimensional datasets. It converts similarities between data points into joint probabilities and minimizes the Kullback-Leibler divergence between the original and reduced distributions.

```python
from sklearn.manifold import TSNE
import numpy as np

data = np.array([[1, 2], [2, 3], [3, 4]])
tsne = TSNE(n_components=2)
reduced_data = tsne.fit_transform(data)
```
### UMAP (Uniform Manifold Approximation and Projection)
UMAP is a dimensionality reduction technique that preserves the global structure of data while allowing for local relationships. It is effective for visualizing complex datasets.
```python
import umap
import numpy as np

data = np.array([[1, 2], [2, 3], [3, 4]])
umap_model = umap.UMAP(n_components=2)
reduced_data = umap_model.fit_transform(data)
```

# Deep Learning

## Neural Networks
### Multilayer Perceptron (MLP): 
A type of feedforward artificial neural network that consists of multiple layers of nodes, where each layer is fully connected to the next. MLPs are capable of learning complex patterns and are widely used for classification and regression tasks.

  ```python
  from tensorflow import keras
  from tensorflow.keras import layers

  model = keras.Sequential([
      layers.Dense(64, activation='relu', input_shape=(input_dim,)),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])
  model.compile(optimizer='adam', loss='mean_squared_error')
```

---

### Convolutional Neural Networks (CNN): 
Specialized neural networks designed for processing structured grid data, such as images. They use convolutional layers to automatically learn spatial hierarchies of features, making them highly effective for image classification and object detection tasks.
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### Recurrent Neural Networks (RNN): 
Designed for sequential data, allowing information to persist across time steps. RNNs are particularly effective for tasks such as time series prediction and natural language processing.
``` python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.SimpleRNN(64, input_shape=(timesteps, features)),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
```

### Long Short-Term Memory (LSTM): 
A type of RNN capable of learning long-term dependencies. It uses special units called memory cells to store information over long periods, making it effective for tasks such as speech recognition and language modeling.
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.LSTM(64, input_shape=(timesteps, features)),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
```
### Gated Recurrent Unit (GRU): 
A variant of LSTM that uses gating mechanisms to control the flow of information. GRUs are simpler and often faster to train than LSTMs while achieving comparable performance on many tasks.
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.GRU(64, input_shape=(timesteps, features)),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
```
### Transformers: 
A type of neural network architecture that relies on self-attention mechanisms to process sequential data. Transformers have become the state-of-the-art for natural language processing tasks, such as translation and text generation.
```python
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(sequence_length, feature_dim))
x = layers.MultiHeadAttention(num_heads=8, key_dim=feature_dim)(inputs, inputs)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
## Loss Functions
### Mean Squared Error (MSE): 
A common loss function used for regression tasks. It measures the average squared difference between predicted and actual values, penalizing larger errors more than smaller ones.
```python
from tensorflow.keras.losses import MeanSquaredError

mse = MeanSquaredError()
loss = mse(y_true, y_pred)
```
### Categorical Cross-Entropy: 
A loss function used for multi-class classification tasks. It measures the dissimilarity between the predicted probability distribution and the true distribution.
```python
from tensorflow.keras.losses import CategoricalCrossentropy

cce = CategoricalCrossentropy()
loss = cce(y_true, y_pred)
```
### Binary Cross-Entropy: 
A loss function used for binary classification tasks. It measures the performance of a model whose output is a probability value between 0 and 1.
```python
from tensorflow.keras.losses import BinaryCrossentropy

bce = BinaryCrossentropy()
loss = bce(y_true, y_pred)
```
# Optimization Algorithms
### Stochastic Gradient Descent (SGD): 
An optimization algorithm used to minimize the loss function by updating model weights iteratively based on the gradient of the loss function. It is widely used due to its simplicity and effectiveness.
```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')
```

### Adam: 
Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm that combines the benefits of both AdaGrad and RMSProp. It is widely used for training deep learning models due to its efficiency and effectiveness.
```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
```
### RMSProp: 
RMSProp is an adaptive learning rate optimization algorithm that adjusts the learning rate for each parameter based on the average of recent magnitudes of the gradients. It is particularly effective for non-stationary objectives.
```python
from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
```

# Model Selection and Evaluation
### Train-Test Split
This technique is used to evaluate the performance of a machine learning model by splitting the dataset into two parts: one for training the model and the other for testing its performance. This helps assess how well the model generalizes to unseen data, reducing the risk of overfitting.
```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
```

### Cross-Validation: 
This technique assesses the generalizability of a model by partitioning the data into multiple subsets. The model is trained on a subset and tested on the remaining data, repeating this process multiple times to obtain a more reliable estimate of model performance. This is particularly useful for small datasets.
```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

data = load_iris()
model = LogisticRegression()
scores = cross_val_score(model, data.data, data.target, cv=5)
```

## Hyperparameter Tuning
### GridSearchCV: 
This method is used to find the optimal hyperparameters for a model by exhaustively searching through a specified parameter grid. It evaluates all combinations of parameters and selects the best-performing set based on cross-validation.
``` python
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
model = RandomForestClassifier()
param_grid = {'n_estimators': [10, 50, 100]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(data.data, data.target)
```

### RandomizedSearchCV: 
This method samples a specified number of parameter combinations from a grid, making it more efficient than GridSearchCV, especially when dealing with a large parameter space.
```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

data = load_iris()
model = RandomForestClassifier()
param_dist = {'n_estimators': randint(10, 100)}
random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5)
random_search.fit(data.data, data.target)
```

# Utility Functions
## Metrics
### accuracy_score: 
This metric measures the proportion of correctly predicted instances out of the total instances. It is a commonly used metric for classification tasks.
```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
accuracy = accuracy_score(y_true, y_pred)
```

### f1_score: 
The F1 Score combines precision and recall into a single score. It is particularly useful when dealing with imbalanced datasets, as it provides a better measure of the incorrectly classified cases.
```python
from sklearn.metrics import f1_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
f1 = f1_score(y_true, y_pred)
```

### precision_score: 
This metric measures the accuracy of positive predictions. It is defined as the ratio of true positive predictions to the total predicted positives.
```python
from sklearn.metrics import precision_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
precision = precision_score(y_true, y_pred)
```

### recall_score: 
Recall measures the ability of a model to identify all relevant instances. It is defined as the ratio of true positive predictions to the total actual positives.
``` python
from sklearn.metrics import recall_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
recall = recall_score(y_true, y_pred)
```
### roc_auc_score: 
ROC AUC Score measures the area under the receiver operating characteristic curve. It provides an aggregate measure of performance across all classification thresholds.
```python
from sklearn.metrics import roc_auc_score

y_true = [0, 1, 1, 0]
y_scores = [0.1, 0.4, 0.35, 0.8]
roc_auc = roc_auc_score(y_true, y_scores)
```

### confusion_matrix: 
Confusion matrix is a table used to evaluate the performance of a classification model. It provides a summary of the correct and incorrect predictions broken down by class.
```python
from sklearn.metrics import confusion_matrix

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
conf_matrix = confusion_matrix(y_true, y_pred)
```











