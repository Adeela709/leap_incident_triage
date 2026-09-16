import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("site_incidents.csv")

bad_values = ["-", "see attached report", "TBC"]

data = data.dropna(subset=["description"])
data = data[~data["description"].isin(bad_values)]

X = data["description"]
y = data["severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer()
model = LogisticRegression()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)

print(data.loc[X_test.index, "incident_id"].to_string())
print(y_pred)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#empty = data[data["description"].isna()]


"""print("Empty descriptions:", len(empty))
print(empty[["incident_id", "description", "severity"]])

print("Training incidents:", len(X_train))
print("Testing incidents:", len(X_test))

print("Missing descriptions:", data["description"].isna().sum())

bad_values = ["", "n/a", "-", "see attached report", "TBC"]

for value in bad_values:
    print(f'"{value}" descriptions:', (data["description"] == value).sum())
"""
# vectoriser converts into numbers and model predicts or makes a decision using those numberss.

new_text = ["worker fell from ladder"]

new_text_tfidf = vectorizer.transform(new_text)

prediction = model.predict(new_text_tfidf)

print(prediction)

probabilities = model.predict_proba(new_text_tfidf)

print(probabilities)

import joblib
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")