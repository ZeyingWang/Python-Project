# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 23:52:50 2021
@author: zeyin
"""

'''import data'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer.keys() #what dictionaries do we have in dataset
Out[26]: dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

print(cancer['DESCR']) #show all the descriptions

cancer['data'].shape
Out[28]: (569, 30)

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'],['target']))

df_cancer.head()



'''Visualizing data'''
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

sns.countplot(df_cancer['target'])

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

plt.figure(figsize = (20,10)) #resize the figure for better visualization
sns.heatmap(df_cancer.corr(), annot = True)



'''Model Training'''
X = df_cancer.drop(['target'], axis = 1) #axis = 1, specify drop the column
y = df_cancer['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 5)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(X_train, y_train)



'''Model Evaluation'''
y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)



'''Data Normalization
1. Feature Scaling (Unity-based normalization) brings all values into range[0,1]
X = (X-Xmin)/(Xmax - Xmin)
2. SVM parameter optimization(Tuning)
C parameter: Control trade-off between classifying training points correctly and having a smooth
decision boundary.
small C(loose) makes cost(penalty) of misclassification low(soft margin);
large C(strict) makes cost(penalty) of misclassification high(hard margin), potentially overfit
small C --> smooth line for boundary, large C --> strict line for boundary
Gamma parameter: Controls how far the influence of a single training set reaches:
    large gamma: close reach(closer data points have higher weight); small gamma: far reach(more
    generalized solution, more points are considered)
'''


'''Model Improvement'''
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train- min_train)/range_train

sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train) 
#Notice here: the y_train is the actual training data without scaling
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train['mean smoothness'], hue = y_train)

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

svc_model.fit(X_train_scaled, y_train) #retrain the model using the scaled data points
y_predict = svc_model.predict(X_test_scaled)




'''imporving the model'''
param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf']} 
#create a grid search to find the optimal parameters
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4) #verbose-> how many values to display

grid.fit(X_train_scaled, y_train) #show all the search results
grid.best_params_ #show the best result

grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, grid_predictions))
