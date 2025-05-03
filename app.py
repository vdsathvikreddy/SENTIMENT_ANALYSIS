import streamlit as st
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sst_2_logistic_regression

# Set page configuration
st.set_page_config(page_title="SST Sentiment Analysis", page_icon="🎬", layout="centered")

# Define base path (relative for Streamlit Cloud)
base_path = os.getcwd()

# Load DistilBERT model and tokenizer
@st.cache_resource
def load_distilbert_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("vdsr/sst5-distilbert")
        model = AutoModelForSequenceClassification.from_pretrained("vdsr/sst5-distilbert")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load DistilBERT model: {e}")
        st.stop()

# Load SST-5 logistic regression model and vectorizer
@st.cache_resource
def load_sst5_logistic_model():
    model_path = os.path.join(base_path, "sst5_logistic_model.joblib")
    vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.joblib")
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("SST-5 logistic regression model or vectorizer not found. Please run sst5.py to train and save the model.")
        st.stop()
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Failed to load SST-5 logistic regression model or vectorizer: {e}")
        st.stop()

# # Load SST-2 logistic regression model and vectorizer
# @st.cache_resource
# def load_sst2_logistic_model():
#     model_path = os.path.join(base_path, "sst2_logistic_model.joblib")
#     vectorizer_path = os.path.join(base_path, "sst2_tfidf_vectorizer.joblib")
#     if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
#         st.error("SST-2 logistic regression model or vectorizer not found. Please run sst_2_logistic_regression.py to train and save the model.")
#         st.stop()
#     try:
#         model = joblib.load(model_path)
#         vectorizer = joblib.load(vectorizer_path)
#         return model, vectorizer
#     except Exception as e:
#         st.error(f"Failed to load SST-2 logistic regression model or vectorizer: {e}")
#         st.stop()

# Label mappings
distilbert_label_map = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}
sst5_logistic_label_map = {
    0: "Negative",
    2: "Neutral",
    4: "Positive"
}
sst2_logistic_label_map = {
    0: "Negative",
    1: "Positive"
}

# Negation handling for SST-5 logistic regression
negation_words = {
    "not", "no", "never", "none", "nobody", "nothing", "neither",
    "nowhere", "hardly", "barely", "scarcely", "isn't", "wasn't",
    "weren't", "doesn't", "don't", "didn't", "won't", "wouldn't",
    "can't", "couldn't", "shouldn't", "cannot"
}

def contains_negation(sentence):
    words = sentence.lower().split()
    return any(word in negation_words for word in words)

def predict_sst5_logistic_sentiment(sentence, model, vectorizer):
    sentence = sentence.lower().strip()
    test = vectorizer.transform([sentence])
    pred = model.predict(test)[0]
    if not contains_negation(sentence):
        if pred == 0:
            out = "Negative"
        elif pred == 2:
            out = "Neutral"
        else:
            out = "Positive"
    else:
        if pred == 0:
            out = "Neutral"
        elif pred == 2:
            out = "Negative"
        else:
            out = "Neutral"
    return out

def predict_sst2_logistic_sentiment(sentence, model, vectorizer):
    s = [sentence]
    test = vectorizer.transform(s)
    pred = model.predict(test)
    if pred == [1]:
        out = "Positive"
    elif pred == [0]:
        out = "Negative"
    return out

# Streamlit UI
st.title("SST Sentiment Analysis")
st.markdown("Enter a movie review to predict its sentiment using a pre-trained DistilBERT (5-class), SST-5 Logistic Regression (3-class), or SST-2 Logistic Regression (2-class) model.")

# Model selection
model_choice = st.selectbox("Select Model", ["DistilBERT", "SST-5 Logistic Regression", "SST-2 Logistic Regression"])

# Load selected model
if model_choice == "DistilBERT":
    tokenizer, model = load_distilbert_model()
    test_accuracy = "~66.9% (on SST-5 test set, 5 classes)"
    label_map = distilbert_label_map
elif model_choice == "SST-5 Logistic Regression":
    model, vectorizer = load_sst5_logistic_model()
    test_accuracy = "64.65% (on SST-5 test set, 3 classes)"
    label_map = sst5_logistic_label_map
else:  # SST-2 Logistic Regression
    # model, vectorizer = load_sst2_logistic_model()
    test_accuracy = "~80% (on SST-2 test set, 2 classes)"
    label_map = sst2_logistic_label_map

# Text input
user_input = st.text_area("Enter your movie review:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip():
        if model_choice == "DistilBERT":
            # Tokenize input
            inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=64)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0].numpy()
            pred = np.argmax(probs)
            sentiment = label_map[pred]
        elif model_choice == "SST-5 Logistic Regression":
            # SST-5 logistic regression prediction with negation
            sentiment = predict_sst5_logistic_sentiment(user_input, model, vectorizer)
        else:
            # SST-2 logistic regression prediction
            sentiment = sst_2_logistic_regression.predict_sentiment(user_input)
        
        # Display results
        st.subheader("Prediction")
        st.write(f"**Sentiment**: {sentiment}")
        
    else:
        st.error("Please enter a valid review.")