import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from tensorflow import keras
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

# broad
classifier = DecisionTreeClassifier()
'''
    steps:
        separate data into resolutions and issues
        vectorize based on passed function (this has one hot encoding right now)
        split into test and train f(resolutions, issues, ...)
        fit classifier
        
    
'''


# data in
# I need to get data and redo these two lines
data = pd.read_csv("/content/drive/MyDrive/MEDTEST.csv") # should just take in one thing of data
updated_data = pd.read_csv("/content/drive/MyDrive/TestDataUpdated.csv")

# labels is the resolutions
# features is the issues (combination of interlocks and error codes)
# separate data into resolutions and issues
updated_data_labels = updated_data['Sub System']  # y label, x data_features
data_features = data.drop('label', axis=1)  # capitalized post one hot encoding
data_features = data_features.fillna(0)
test_data_label = data['label']


# Perform cell 23 (from colab) with subsystem data
one_hot_test_data_features = pd.get_dummies(data_features)  # Was same for both

updated_data_labels = pd.get_dummies(updated_data_labels)
one_hot_split_train_features, one_hot_split_test_features, one_hot_split_train, one_hot_split_test = train_test_split(one_hot_test_data_features, updated_data_labels, test_size=0.756, random_state=1)
print(one_hot_split_test.shape)

clft = classifier.fit(one_hot_split_train_features, one_hot_split_train)
t_pred = clft.predict(one_hot_split_test_features)  # t means has been split into train and test
print("Accuracy of updated test data:", metrics.accuracy_score(one_hot_split_test, t_pred))

# Perform one hot encoding and train and test decision tree on one hot encodings
test_data_label = pd.get_dummies(test_data_label)
X_train, X_test, y_train, y_test = train_test_split(one_hot_test_data_features, test_data_label, test_size=0.3, random_state=42)
classifier = classifier.fit(X_train, y_train)  # Train Decision Tree Classifier
y_pred = classifier.predict(X_test)  # Predict the response for test dataset
print(y_test.shape)
print(y_pred.shape)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

def one_hot_vectorizer ():
    #  TODO function that vectorizes based on ont hot encoding

def vectorize_data ():
    """

    :return:
    """