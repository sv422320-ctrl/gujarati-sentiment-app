import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from model_BERT_CNN import BertCNN   # we'll create this wrapper

# Load model + tokenizer
MODEL_NAME = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

@st.cache_resource
def load_model():
    model = BertCNN(model_name=MODEL_NAME, num_labels=3)  # 3 classes: pos/neg/neutral
    model.load_state_dict(torch.load("model/bert_cnn_gujarati.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        pred = torch.argmax(logits, dim=1).item()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[pred]

# Streamlit UI
st.title("📊 Gujarati Sentiment Analysis (BERT+CNN)")
st.write("Enter Gujarati text or upload a CSV file to analyze sentiment.")

# Text input
text_input = st.text_area("✍️ Enter Gujarati Sentence:")
if st.button("Analyze Sentence"):
    if text_input.strip():
        result = predict_sentiment(text_input)
        st.success(f"Prediction: **{result}**")
    else:
        st.warning("Please enter some text.")

# CSV upload
uploaded_file = st.file_uploader("📂 Upload a CSV file with a 'Sentence' column", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "Sentence" in df.columns:
        df["Predicted Sentiment"] = df["Sentence"].apply(predict_sentiment)
        st.dataframe(df)
        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
    else:
        st.error("CSV must contain a 'Sentence' column.")
