import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sst_2_logistic_regression
import sst5

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("vdsr/sst5-distilbert")
    model = AutoModelForSequenceClassification.from_pretrained("vdsr/sst5-distilbert")
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit app
st.title("SENTIMENT CLASSIFICATION")
st.write("Enter a sentence to predict its sentiment ( Negative, Neutral, Positive).")

# Input text
user_input = st.text_input("Enter a sentence to classify:", "")

if st.button("Predict Using sst2 trained logistic regression"):
    out = sst_2_logistic_regression.predict_sentiment(user_input)
    st.success(f"**{out}**")

if st.button("Predict Using sst5 trained logistic regression"):
    out = sst5.predict_sentiment(user_input)
    st.success(f"**{out}**")

# Predict sentiment
if st.button("Predict using distilbert"):
    if user_input.strip():
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        label_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
        sentiment = label_map[prediction]
        
        st.success(f"**{sentiment}**")
    else:
        st.error("Please enter a valid sentence.")


