import streamlit as st
import joblib
import re

# load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

labels = {
    0: "anger",
    1: "joy",
    2: "fear",
    3: "sadness",
    4: "love",
    5: "surprise"
}

def clean_text(t):
    t = t.lower()
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t

st.title("Sentiment Analysis App")
text = st.text_area("Enter a sentence in English:")

if st.button("Predict"):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    st.success(f"Predicted emotion: {labels[pred]}")
