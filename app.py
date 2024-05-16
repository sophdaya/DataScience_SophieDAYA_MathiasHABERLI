import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import numpy as np

# Load the tokenizer and model
model_path = "./model"
tokenizer = CamembertTokenizer.from_pretrained(model_path)
model = CamembertForSequenceClassification.from_pretrained(model_path)

# Define a function to predict the difficulty
def predict_difficulty(text):
    # Tokenize the input text
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
    
    # Ensure the model is in evaluation mode and make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    return levels[predicted_class_id]

# Streamlit app interface
st.title("French Text Difficulty Predictor")
st.write("Enter a French text and get its difficulty level predicted.")

# Text input
user_input = st.text_area("Enter French text here:", "")

# Predict button
if st.button("Predict Difficulty"):
    if user_input.strip() == "":
        st.write("Please enter a valid French text.")
    else:
        # Predict difficulty
        difficulty = predict_difficulty(user_input)
        st.write(f"The predicted difficulty level is: **{difficulty}**")
