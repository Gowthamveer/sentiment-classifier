
import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

st.set_page_config(page_title="Sentiment Classifier", page_icon="🎬", layout="centered")

@st.cache_resource
def load_model():
    MODEL_PATH = "/content/drive/MyDrive/sentiment-classifier/models"
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

st.title("🎬 Sentiment Classifier")
st.markdown("Powered by **DistilBERT** fine-tuned on IMDB reviews")
st.markdown("---")

text = st.text_area("Enter a movie review or any text:", height=150,
                     placeholder="e.g. This movie was absolutely fantastic!")

if st.button("Analyze Sentiment 🔍"):
    if text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=256
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item() * 100

        if pred == 1:
            st.success(f"### Positive 😊")
        else:
            st.error(f"### Negative 😞")

        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(confidence / 100)

        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric("Negative", f"{probs[0][0].item()*100:.2f}%")
        col2.metric("Positive", f"{probs[0][1].item()*100:.2f}%")
