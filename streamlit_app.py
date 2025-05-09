import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
from io import BytesIO
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForQuestionAnswering
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import string

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
    def remove_articles(text):
        return re.sub(r'\\b(a|an|the)\\b', ' ', text)
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

def generate_true_answer(question, blip_answer, vilt_answer):
    """
    Heuristic true answer generation:
    - If BLIP and ViLT agree, take that as true answer.
    - If not, randomly choose one of them or a plausible default.
    This simulates automated 'true' answer creation in absence of dataset.
    """
    blip_norm = normalize_answer(blip_answer)
    vilt_norm = normalize_answer(vilt_answer)
    if blip_norm == vilt_norm:
        return blip_norm
    # 50% chance pick BLIP or ViLT answer, else default plausible answer
    fallback_answers = ["a dog", "a cat", "a car", "a tree", "a house"]
    choices = [blip_norm, vilt_norm] + fallback_answers
    return random.choice(choices)

# Widgets setup
upload_widget = widgets.FileUpload(accept='.jpg,.png', multiple=False)
question_widget = widgets.Text(placeholder="Enter your question here")
model_dropdown = widgets.Dropdown(options=["BLIP", "ViLT", "Both", "Ensemble"], value="Both", description="Model:")
submit_button = widgets.Button(description="Submit")
image_output = widgets.Output()
response_output = widgets.Output()

display(upload_widget, image_output, question_widget, model_dropdown, submit_button, response_output)

uploaded_image = None

def on_image_upload(change):
    global uploaded_image
    image_output.clear_output()
    if upload_widget.value:
        file_info = next(iter(upload_widget.value.values()))
        uploaded_image = Image.open(BytesIO(file_info['content'])).convert("RGB")
        with image_output:
            display(uploaded_image)

upload_widget.observe(on_image_upload, names='value')

def ensemble_answer(blip_answer, vilt_answer):
    # Return majority vote or fallback if no agreement
    blip_norm = normalize_answer(blip_answer)
    vilt_norm = normalize_answer(vilt_answer)
    if blip_norm == vilt_norm:
        return blip_norm
    # If disagree, choose by simple priority or randomly
    return blip_norm  # For example, prioritize BLIP; you can randomize or use confidence if available

def plot_confidence(labels, probs, title, highlight=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels[::-1], probs[::-1], color='lightsteelblue', edgecolor='black')
    if highlight:
        for bar, label in zip(bars, labels[::-1]):
            if normalize_answer(label) == normalize_answer(highlight):
                bar.set_color('green')
    ax.set_xlabel("Confidence")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def on_submit(b):
    global uploaded_image
    with response_output:
        response_output.clear_output()
        if uploaded_image is None:
            print("⚠️ Please upload an image.")
            return
        if not question_widget.value.strip():
            print("⚠️ Please enter a question.")
            return
        image = uploaded_image
        question = question_widget.value.strip()
        selected_model = model_dropdown.value

        blip_answer = None
        vilt_answer = None
        true_answer = None

        # Get predictions from models as needed
        if selected_model in ["BLIP", "Both", "Ensemble"]:
            blip_answer = get_blip_answer(image, question)
            print(f"🔵 BLIP Answer: {blip_answer}")

        if selected_model in ["ViLT", "Both", "Ensemble"]:
            vilt_answer, top5_labels, top5_probs = get_vilt_answer(image, question)
            print(f"🟢 ViLT Answer: {vilt_answer}")
            # Show ViLT confidence bar chart
            plot_confidence(top5_labels, top5_probs, "ViLT Top-5 Prediction Confidence", highlight=vilt_answer)

        if selected_model == "BLIP":
            # Since BLIP provides one answer without confidence, simulate simple confidence display
            if blip_answer:
                plot_confidence([blip_answer], [1.0], "BLIP Confidence", highlight=blip_answer)

        if selected_model == "Both":
            if blip_answer:
                plot_confidence([blip_answer], [1.0], "BLIP Confidence", highlight=blip_answer)

        # Generate true answer heuristically if ensemble or Both selected
        if selected_model in ["Ensemble","Both"]:
            true_answer = generate_true_answer(question, blip_answer or "", vilt_answer or "")
            print(f"🎯 Generated True Answer (heuristic): {true_answer}")
        else:
            # For BLIP or ViLT only mode, just generate true answer based on that model answer, simplified.
            true_answer = generate_true_answer(question, blip_answer or "", vilt_answer or "")

        # Prepare lists for metric calculation (single prediction example)
        true_answers = [normalize_answer(true_answer)]
        blip_preds = [normalize_answer(blip_answer)] if blip_answer else []
        vilt_preds = [normalize_answer(vilt_answer)] if vilt_answer else []

        # Ensemble prediction and metrics
        if selected_model == "Ensemble":
            ensembled = ensemble_answer(blip_answer or "", vilt_answer or "")
            print(f"⚫ Ensemble Answer: {ensembled}")
            ensemble_preds = [normalize_answer(ensembled)]

        # Calculate and show metrics
        def calc_and_print_metrics(name, y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            print(f"\n📊 {name} Metrics:")
            print(f" Accuracy:  {acc:.2f}")
            print(f" Precision: {prec:.2f}")
            print(f" Recall:    {rec:.2f}")
            print(f" F1 Score:  {f1:.2f}")

        if blip_preds:
            calc_and_print_metrics("BLIP", true_answers, blip_preds)
        if vilt_preds:
            calc_and_print_metrics("ViLT", true_answers, vilt_preds)
        if selected_model == "Ensemble":
            calc_and_print_metrics("Ensemble", true_answers, ensemble_preds)

submit_button.on_click(on_submit)

        
