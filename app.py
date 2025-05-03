import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("vdsr/sst5-distilbert")
    model = AutoModelForSequenceClassification.from_pretrained("vdsr/sst5-distilbert")
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit app
st.title("SST-5 Sentiment Analysis with DistilBERT")
st.write("Enter a sentence to predict its sentiment (Very Negative, Negative, Neutral, Positive, Very Positive).")
st.write("**Test Accuracy**: ~66.9% (trained on 10,000 phrases, evaluated on 2,000 phrases, 1 epoch)")

# Input text
user_input = st.text_input("Enter a sentence:", "This movie is not good")

# Predict sentiment
if st.button("Predict"):
    if user_input.strip():
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        label_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
        sentiment = label_map[prediction]
        
        st.success(f"Predicted sentiment: **{sentiment}**")
    else:
        st.error("Please enter a valid sentence.")

# Example phrases
st.write("Example predictions:")
test_phrases = ["This movie is not good", "Absolutely fantastic film", "It was okay"]
for phrase in test_phrases:
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    label_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    st.write(f"Phrase: '{phrase}' → Sentiment: {label_map[prediction]}")