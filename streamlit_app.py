import streamlit as st
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForQuestionAnswering
import matplotlib.pyplot as plt

st.set_page_config(page_title="Visual Question Answering", layout="centered")

# Load models once
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda" if torch.cuda.is_available() else "cpu")

    vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to("cuda" if torch.cuda.is_available() else "cpu")

    return blip_processor, blip_model, vilt_processor, vilt_model

# Load models
blip_processor, blip_model, vilt_processor, vilt_model = load_models()
device = "cuda" if torch.cuda.is_available() else "cpu"

# BLIP
def get_blip_answer(image, question):
    inputs = blip_processor(image, question, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

# ViLT
def get_vilt_answer(image, question):
    inputs = vilt_processor(image, question, return_tensors="pt").to(device)
    outputs = vilt_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    top5 = torch.topk(probs, 5)
    top5_labels = [vilt_model.config.id2label[idx.item()] for idx in top5.indices]
    top5_probs = top5.values.detach().cpu().numpy()
    predicted_answer = vilt_model.config.id2label[probs.argmax().item()]
    return predicted_answer, top5_labels, top5_probs

# UI
st.title("🖼️ Visual Question Answering")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display uploaded image immediately
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Input question and model selection
question = st.text_input("Enter your question:")
model_choice = st.selectbox("Choose a model:", ["BLIP", "ViLT", "Both"])

# Submit
if st.button("Submit"):
    if uploaded_file and question:
        if model_choice in ["BLIP", "Both"]:
            with st.spinner("Getting answer from BLIP..."):
                blip_answer = get_blip_answer(image, question)
                st.success(f"🔵 **BLIP Answer:** {blip_answer}")

        if model_choice in ["ViLT", "Both"]:
            with st.spinner("Getting answer from ViLT..."):
                vilt_answer, top5_labels, top5_probs = get_vilt_answer(image, question)
                st.success(f"🟢 **ViLT Answer:** {vilt_answer}")

                # Confidence bar chart
                st.write("### ViLT Top-5 Predictions")
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.barh(top5_labels[::-1], top5_probs[::-1], color="skyblue")
                for bar, label in zip(bars, top5_labels[::-1]):
                    if label == vilt_answer:
                        bar.set_color("green")
                ax.set_xlabel("Confidence")
                ax.set_title("ViLT Prediction Confidence")
                st.pyplot(fig)
    else:
        st.warning("Please upload an image and enter a question.")
