import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
st.set_page_config(page_title="AI Spam Detector Pro", page_icon="🛡️")
st.title("🛡️ Email Spam Detection System")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    return text
with st.sidebar:
    st.header("Upload Area")
    uploaded_file = st.file_uploader("Upload your spam_dataset.csv", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["email_text"] = df["email_text"].apply(clean_text)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=10000)
    X = vectorizer.fit_transform(df["email_text"])
    y = df["label"]
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X, y)
    st.success("Model is Ready! ✅")
    st.divider()
    user_input = st.text_area("Paste an email here to test:")
    if st.button("Analyze Now"):
        if user_input:
            vec_input = vectorizer.transform([clean_text(user_input)])
            prediction = model.predict(vec_input)[0]
            if prediction == 1:
                st.error("🚨 Result: This is a SPAM email!")
            else:
                st.success("✅ Result: This is a SAFE (Ham) email.")
else:
    st.info("Please upload the dataset from the sidebar to start.")
