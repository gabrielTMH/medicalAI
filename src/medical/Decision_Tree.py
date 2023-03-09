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
    #so I think a problem with this function is that most vectorizers need to be fit on the data and
    #if we wanted the data we are passing in to be vectorized it would be with a transform method
    return vectorizer(issues, resolutions)


def one_hot_vectorizer(issues, resolutions):
    return pd.get_dummies(issues), pd.get_dummies(resolutions)


def tfidf_vectorizer(data, issues, resolutions):
    #as much as I appreciate this method I think we use it in the pipeline
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


def predict_response(issues_train, issues_test, resolutions_train, resolutions_test, classifier, print_results):
    classifier = classifier()
    fitted_classifier = classifier.fit(issues_train, resolutions_train)
    prediction = fitted_classifier.predict(issues_test)
    print_results(resolutions_test, prediction)


def one_hot_classifier():
    #not sure why we have this class also its call one-hot but its just returning a decision tree
    return DecisionTreeClassifier()

def tfidf_DF_pipeline():
    return Pipeline([
        ('vectorizer_tfidf', TfidfVectorizer()),
        ('random_forest', RandomForestClassifier())
    ])

def tfidf_MLP_pipeline():
    return Pipeline([
        ('vectorizer_tfidf', TfidfVectorizer()),
        ('random_forest', MLPClassifier(hidden_layer_sizes=(500,), max_iter=1000, random_state=42))
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





# class DecisionTree:
#     #I think there is an argument to made about how issues and resolutions is kinda long and could be replaced with x and y
#     #data = pd.read_csv("reorganized.csv") put this line in prep_data
#     issues, resolutions = prep_data("reorganized.csv")
#     vectorized_issues, vectorized_resolutions = vectorize_data(tfidf_vectorizer(), issues, resolutions) #this line needs reworking
#     issues_train, issues_test, resolutions_train, resolutions_test = \
#         train_test_split(vectorized_issues, vectorized_resolutions, test_size=0.3, random_state=1)
#     predict_response(issues_train, issues_test, resolutions_train,
#                      resolutions_test, tfidf_KNN_pipeline(), tfidf_print_results())





# data = pd.read_csv("reorganized.csv") # whatever our data thing is
# issues, resolutions = prep_data("reorganized.csv")
# vectorized_issues, vectorized_resolutions = vectorize_data(tfidf_vectorizer(), issues, resolutions) #this is not the line to use
# print(vectorized_issues)
# issues_train, issues_test, resolutions_train, resolutions_test = \
#     train_test_split(vectorized_issues, vectorized_resolutions, test_size=0.3, random_state=1)
# pipeline = tfidf_KNN_pipeline()
# pipeline.fit(issues_train, resolutions_train)
# pickle.dump(pipeline, open('pipeline.pkl', 'wb'))
# pickled_pipeline = pickle.load('pipeline.pkl', 'rb')


def train_and_pickle_pipeline(filename="reorganized.csv", pipeline=tfidf_KNN_pipeline(), path='pipeline.pkl'):
    fileData = pd.read_csv(filename)
    issues, resolutions = prep_data(fileData)
    x_train, x_test, y_train, y_test = train_test_split(issues,resolutions, test_size=0.3, random_state=1)
    pipeline.fit(x_train, y_train)
    pickle.dump(pipeline, open(path, 'wb'))


pipe = tfidf_DF_pipeline()
x_data,y_data = prep_data("reorganized.csv")
shuffle(x_data, y_data, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.4, random_state=1)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
bal_accuracy = balanced_accuracy_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(bal_accuracy)