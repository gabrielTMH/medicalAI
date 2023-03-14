import pickle
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.utils import shuffle
from sklearn.neural_network import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def prep_data(filename):
    data = pd.read_csv(filename)
    return data['issue'].fillna(0), data['resolution']

def vectorize_data(vectorizer, issues, resolutions):
    return vectorizer.fit_transform(issues), vectorizer.fit_transform(resolutions)

def one_hot_vectorizer(issues, resolutions):
    return pd.get_dummies(issues), pd.get_dummies(resolutions)

def tfidf_vectorizer():
    return TfidfVectorizer()

def predict_response(issues_train, issues_test, resolutions_train, resolutions_test, classifier, print_results):
    classifier = classifier()
    fitted_classifier = classifier.fit(issues_train, resolutions_train)
    prediction = fitted_classifier.predict(issues_test)
    print_results(resolutions_test, prediction)

def one_hot_classifier():
    return DecisionTreeClassifier()

def tfidf_DF_pipeline():
    return Pipeline([
        ('vectorizer_tfidf', TfidfVectorizer()),
        ('random_forest', RandomForestClassifier())
    ])

def tfidf_MLP_pipeline():
    return Pipeline([
        ('vectorizer_tfidf', TfidfVectorizer()),
        ('MLP', MLPClassifier(hidden_layer_sizes=(500,), max_iter=1000, random_state=42))
    ])

def tfidf_MNB_pipeline():
    return Pipeline([
        ('vectorizer_tfidf', TfidfVectorizer()),
        ('random_forest', MultinomialNB())
    ])

def tfidf_KNN_pipeline():
    return Pipeline([
        ('vectorizer_tfidf', TfidfVectorizer()),
        ('KNN', KNeighborsClassifier())
    ])

def one_hot_print_results(resolutions_test, prediction):
    print("Accuracy: ", metrics.accuracy_score(resolutions_test, prediction))

def tfidf_print_results(resolutions_test, prediction):
    print(classification_report(resolutions_test, prediction))

def train_and_pickle_pipeline(filename, pipeline, path):
    file_data = pd.read_csv(filename)
    issues, resolutions = prep_data(file_data)
    x_train, x_test, y_train, y_test = train_test_split(issues, resolutions, test_size=0.3, random_state=1)
    pipeline.fit(x_train, y_train)
    pickle.dump(pipeline, open(path, 'wb'))

# Example usage:
if __name__ == '__main__':
    file_path = "/Users/cristianpanaro/PycharmProjects/sofdev-s23-medical/data/TestDataUpdated.csv"
    pipeline = tfidf_KNN_pipeline()
    pipeline_path = 'pipeline.pkl'
    train_and_pickle_pipeline(file_path, pipeline, pipeline_path)
