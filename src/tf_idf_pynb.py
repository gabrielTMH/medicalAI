from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


v = TfidfVectorizer()
df = pd.read_csv("../../../../Desktop/reorganized.csv")

has_seen = {}
count = 0
row_num = 0
for label in df.label:
    if str(label) not in has_seen:
        df.loc[row_num, 'label_num'] = count
        has_seen[label] = count
        count += 1
    else:
        df.loc[row_num, 'label_num'] = has_seen[label]
    row_num += 1

X_train, X_test, y_train, y_test = train_test_split(
    df.issue,
    df.label_num,
    test_size=0.4,
    random_state=2022,
    stratify=df.label_num
)

clf = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer()),
    ('KNN', KNeighborsClassifier())
])

# train model
clf.fit(X_train, y_train)
pickle.dump(clf,'dummy.path')
# prediction
y_pred = clf.predict(X_test)

# compare
print(classification_report(y_test, y_pred))
