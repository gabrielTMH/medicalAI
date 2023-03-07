import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


def Decision_Tree(input_data):
    issues, resolutions = prep_data(input_data)
    vectorized_issues, vectorized_resolutions = vectorize_data(one_hot_vectorizer, issues, resolutions)
    issues_train, issues_test, resolutions_train, resolutions_test = \
        train_test_split(vectorized_issues, vectorized_resolutions, test_size=0.3, random_state=1)
    predict_response(issues_train, issues_test, resolutions_train, resolutions_test)


def prep_data(input_data):
    return input_data['issue'].fillna(0), input_data['resolution']


def vectorize_data(vectorizer, issues, resolutions):
    return vectorizer(issues, resolutions)


def predict_response(issues_train, issues_test, resolutions_train, resolutions_test):
    classifier = DecisionTreeClassifier()
    fitted_classifier = classifier.fit(issues_train, resolutions_train)
    prediction = fitted_classifier.predict(issues_test)
    print("Accuracy: ", metrics.accuracy_score(resolutions_test, prediction))


def one_hot_vectorizer(issues, resolutions):
    return pd.get_dummies(issues), pd.get_dummies(resolutions)


def tfidf_vectorizer(data, issues, resolutions):
    has_seen = {}
    count = 0
    row_num = 0
    for resolution in resolutions:
        if str(resolution) not in has_seen:
            data.loc[row_num, 'label_num'] = count
            has_seen[resolution] = count
            count += 1
        else:
            data.loc[row_num, 'label_num'] = has_seen[resolution]
        row_num += 1

def pickling_data(data):

