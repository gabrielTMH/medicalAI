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
from sklearn.model_selection import GridSearchCV


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
        ('vectorizer', TfidfVectorizer()),
        ('classifier', RandomForestClassifier(bootstrap=True, max_depth=10, min_samples_leaf=1,min_samples_split=1,n_estimators=10))
    ])

def tfidf_MLP_pipeline():
    return Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MLPClassifier(hidden_layer_sizes=(500,), max_iter=1000, random_state=42))
    ])

def tfidf_MNB_pipeline():
    return Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])

def tfidf_KNN_pipeline():
    return Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', KNeighborsClassifier())
    ])

def one_hot_print_results(resolutions_test, prediction):
    print("Accuracy: ", metrics.accuracy_score(resolutions_test, prediction))

def tfidf_print_results(resolutions_test, prediction):
    print(classification_report(resolutions_test, prediction))


def tune_hyperParamters():
    pipe=tfidf_DF_pipeline()
    param_grid = {
        'n_estimators': [10,25, 50,100],
        'max_depth': [2,5, 10, 20,100],
        'min_samples_split': [1,5,10, 20],
        'min_samples_leaf': [1,3,5,10]
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    issues, resolutions = prep_data("reorganized.csv")
    x_train, x_test, y_train, y_test = train_test_split(issues,resolutions, test_size=0.3, random_state=1)
    pipe.fit(x_train, y_train)
    x_train=pipe['vectorizer'].transform(x_train)
    grid_search.fit(x_train, y_train)
    print('Best parameters:', grid_search.best_params_)
    print('Accuracy:', grid_search.best_score_)

def train_and_pickle_pipeline(filename="reorganized.csv", pipeline=tfidf_DF_pipeline(), path='pipeline.pkl',rs=1):
    issues, resolutions = prep_data(filename)
    x_train, x_test, y_train, y_test = train_test_split(issues,resolutions, test_size=0.3, random_state=rs)
    pipeline.fit(x_train, y_train)
    pickle.dump(pipeline, open(path, 'wb'))


# Example usage:
if __name__ == '__main__':
    train_and_pickle_pipeline()
    #tune_hyperParamters()