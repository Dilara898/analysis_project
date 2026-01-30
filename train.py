import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


train_df = pd.read_csv("training.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

print(train_df.head())
print(train_df["label"].value_counts())


def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"\d+", "", t)          # remove numbers
    t = re.sub(r"[^\w\s]", "", t)      # remove punctuation/symbols
    t = re.sub(r"\s+", " ", t).strip() # remove extra spaces
    return t

train_df["text"] = train_df["text"].apply(clean_text)
val_df["text"]   = val_df["text"].apply(clean_text)
test_df["text"]  = test_df["text"].apply(clean_text)

print(train_df["text"].head())


vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_df["text"])

X_val = vectorizer.transform(val_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

print("Validation accuracy:", accuracy_score(y_val, val_pred))
print("Test accuracy:", accuracy_score(y_test, test_pred))

print("\nClassification report:")
print(classification_report(y_test, test_pred))


joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")