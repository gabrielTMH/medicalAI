import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


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


def predict_response(issues_train, issues_test, resolutions_train, resolutions_test, classifier, print_results):
    classifier = classifier()
    fitted_classifier = classifier.fit(issues_train, resolutions_train)
    prediction = fitted_classifier.predict(issues_test)
    print_results(resolutions_test, prediction)


def one_hot_classifier():
    return DecisionTreeClassifier()


def tfidf_classifier():
    classifier = Pipeline([
        ('vectorizer_tfidf', TfidfVectorizer()),
        ('KNN', KNeighborsClassifier())
    ])
    return classifier


def one_hot_print_results(resolutions_test, prediction):
    print("Accuracy: ", metrics.accuracy_score(resolutions_test, prediction))


def tfidf_print_results(resolutions_test, prediction):
    print(classification_report(resolutions_test, prediction))


def one_hot_vectorizer(issues, resolutions):
    return pd.get_dummies(issues), pd.get_dummies(resolutions)


def tfidf_vectorizer(data, issues, resolutions):
    has_seen = {}
    count = 0
    row_num = 0
    for resolution in resolutions:
        if str(resolution) not in has_seen:
            data.loc[row_num, 'resolution_num'] = count
            has_seen[resolution] = count
            count += 1
        else:
            data.loc[row_num, 'resolution_num'] = has_seen[resolution]
        row_num += 1