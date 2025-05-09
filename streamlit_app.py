import streamlit as st
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForQuestionAnswering
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)
    return blip_processor, blip_model, vilt_processor, vilt_model

blip_processor, blip_model, vilt_processor, vilt_model = load_models()

# Utility Functions
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    def white_space_fix(text):
        return ' '.join(text.split())
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def get_blip_answer(image, question):
    inputs = blip_processor(image, question, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs)
    answer = blip_processor.decode(output[0], skip_special_tokens=True)
    return answer

def get_vilt_answer(image, question):
    inputs = vilt_processor(image, question, return_tensors="pt").to(device)
    outputs = vilt_model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
    top5 = torch.topk(probs, 5)
    top5_probs = top5.values.detach().cpu().numpy()
    top5_labels = [vilt_model.config.id2label[idx.item()] for idx in top5.indices]
    predicted_answer = vilt_model.config.id2label[probs.argmax().item()]
    return predicted_answer, top5_labels, top5_probs

def ensemble_answer(blip_answer, vilt_answer):
    blip_norm = normalize_answer(blip_answer)
    vilt_norm = normalize_answer(vilt_answer)
    if blip_norm == vilt_norm:
        return blip_norm
    return blip_norm  # prioritize BLIP

def generate_true_answer(question, blip_answer, vilt_answer):
    blip_norm = normalize_answer(blip_answer)
    vilt_norm = normalize_answer(vilt_answer)
    if blip_norm == vilt_norm:
        return blip_norm
    fallback_answers = ["a dog", "a cat", "a car", "a tree", "a house"]
    choices = [blip_norm, vilt_norm] + fallback_answers
    return random.choice(choices)

def plot_confidence(labels, probs, title, highlight=None):
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(labels[::-1], probs[::-1], color='lightsteelblue', edgecolor='black')
    if highlight:
        for bar, label in zip(bars, labels[::-1]):
            if normalize_answer(label) == normalize_answer(highlight):
                bar.set_color('green')
    ax.set_xlabel("Confidence")
    ax.set_title(title)
    st.pyplot(fig)

def calc_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    st.markdown(f"### 📊 {name} Metrics")
    st.write(f"**Accuracy**: {acc:.2f}")
    st.write(f"**Precision**: {prec:.2f}")
    st.write(f"**Recall**: {rec:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")

# Streamlit UI
st.set_page_config(page_title="VQA: BLIP vs ViLT", layout="wide")
st.title("🔍 Visual Question Answering: BLIP vs ViLT")

uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])
question = st.text_input("💬 Enter your question:")
selected_model = st.selectbox("🧠 Choose a model", ["BLIP", "ViLT", "Both", "Ensemble"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

if st.button("🔎 Submit"):
    if uploaded_file is None or not question.strip():
        st.warning("⚠️ Please upload an image and enter a question.")
    else:
        blip_answer = vilt_answer = true_answer = None
        blip_preds = vilt_preds = []

        if selected_model in ["BLIP", "Both", "Ensemble"]:
            blip_answer = get_blip_answer(image, question)
            st.success(f"🔵 **BLIP Answer:** {blip_answer}")
            plot_confidence([blip_answer], [1.0], "BLIP Confidence", highlight=blip_answer)

        if selected_model in ["ViLT", "Both", "Ensemble"]:
            vilt_answer, top5_labels, top5_probs = get_vilt_answer(image, question)
            st.success(f"🟢 **ViLT Answer:** {vilt_answer}")
            plot_confidence(top5_labels, top5_probs, "ViLT Top-5 Confidence", highlight=vilt_answer)

        if selected_model in ["Both", "Ensemble"]:
            true_answer = generate_true_answer(question, blip_answer or "", vilt_answer or "")
            st.info(f"🎯 **Generated True Answer (heuristic):** {true_answer}")
        else:
            true_answer = generate_true_answer(question, blip_answer or "", vilt_answer or "")

        true_answers = [normalize_answer(true_answer)]
        if blip_answer:
            blip_preds = [normalize_answer(blip_answer)]
        if vilt_answer:
            vilt_preds = [normalize_answer(vilt_answer)]

        if selected_model == "Ensemble":
            ensemble_pred = [normalize_answer(ensemble_answer(blip_answer or "", vilt_answer or ""))]
            st.success(f"⚫ **Ensemble Answer:** {ensemble_pred[0]}")

        if blip_preds:
            calc_metrics("BLIP", true_answers, blip_preds)
        if vilt_preds:
            calc_metrics("ViLT", true_answers, vilt_preds)
        if selected_model == "Ensemble":
            calc_metrics("Ensemble", true_answers, ensemble_pred)
