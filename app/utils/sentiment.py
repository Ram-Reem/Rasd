
import os
import json
import re
import torch
import pandas as pd
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "./sentiment_model"

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))
    with open(os.path.join(MODEL_DIR, "label_mapping.json"), "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    model = torch.jit.load(os.path.join(MODEL_DIR, "sentiment_model.pt"), map_location=DEVICE)
    model.eval()
    return model, tokenizer, label_mapping, inv_label_mapping

arabic_model, arabic_tokenizer, arabic_label_mapping, arabic_inv_label_mapping = load_model_and_tokenizer()

ENG_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
eng_tokenizer = AutoTokenizer.from_pretrained(ENG_MODEL_NAME)
eng_model = AutoModelForSequenceClassification.from_pretrained(ENG_MODEL_NAME).to(DEVICE)
eng_labels = ["Negative", "Neutral", "Positive"]



def is_english(text: str) -> bool:
    english_chars = re.findall(r'[A-Za-z]', text)
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    return len(english_chars) > len(arabic_chars)



def analyze_sentiment(texts: List[str]) -> dict:
    results = []
    counts = {label: 0 for label in arabic_label_mapping.keys()}  # نفس تسميات كامل بيرت

    eng_to_common = {
        "Negative": "negative",
        "Neutral": "neutral",
        "Positive": "positive"
    }

    for text in texts:
        if is_english(text):
            
            inputs = eng_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = eng_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            label_eng = eng_labels[pred]         
            label = eng_to_common[label_eng]     
        else:
            
            inputs = arabic_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            with torch.no_grad():
                logits = arabic_model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            label = arabic_inv_label_mapping.get(pred_class, "Unknown")

        counts[label] += 1
        results.append({"text": text, "label": label})

    return {"counts": counts, "classified_comments": results}

# Extract the top 3 employees based only on "هل تم حل المشكلة" and "هل انت راضي عن الحل"
def extract_top_employees(df: pd.DataFrame) -> Optional[List[dict]]:
    if "اسم المسنده اليه" not in df.columns:
        return None
    if "هل تم حل المشكلة" not in df.columns or "هل انت راضي عن الحل" not in df.columns:
        return None

    
    mask = (
        (df["هل تم حل المشكلة"].astype(str).str.lower() == "y") &
        (df["هل انت راضي عن الحل"].astype(str).str.lower() == "y")
    )

    top_employees = (
        df[mask]
        .groupby("اسم المسنده اليه")
        .size()
        .sort_values(ascending=False)
        .head(3)
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    return top_employees


