import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from tensorflow import keras
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


pima = pd.read_csv("/content/drive/MyDrive/MEDTEST.csv")
toto = pd.read_csv("/content/drive/MyDrive/TestDataUpdated.csv")
y_toto =toto['Sub System']
x_data=pima.drop('label', axis=1)
x_data=x_data.fillna(0)
y_data=pima['label']
print(x_data)

# preform cell 23 with subsystem data
X_data = pd.get_dummies(x_data)
clf = DecisionTreeClassifier()
y_toto = pd.get_dummies(y_toto)
Xt_train, Xt_test, t_train, t_test = train_test_split(X_data, y_toto, test_size=.756, random_state=1)
print(t_test.shape)

clft = clf.fit(Xt_train,t_train)
t_pred = clft.predict(Xt_test)
print("Accuracy of toto:",metrics.accuracy_score(t_test, t_pred))

# preform one hot encoding and train and test decision tree on one hot encodings
X_data=pd.get_dummies(x_data)
y_data=pd.get_dummies(y_data)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
# Predict the response for test dataset
y_pred = clf.predict(X_test)
print(y_test.shape)
print(y_pred.shape)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))