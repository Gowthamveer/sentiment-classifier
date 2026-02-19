
import gradio as gr
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model
model_name = "GowthamVeer45/sentiment-classifier"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def predict_sentiment(text):
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

    label = "Positive 😊" if pred == 1 else "Negative 😞"
    result = {
        "Sentiment": label,
        "Confidence": f"{confidence:.2f}%",
        "Positive Score": f"{probs[0][1].item()*100:.2f}%",
        "Negative Score": f"{probs[0][0].item()*100:.2f}%"
    }
    return result

# Gradio UI
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter a movie review or any text here...",
        label="Input Text"
    ),
    outputs=gr.JSON(label="Results"),
    title="🎬 Sentiment Classifier",
    description="Fine-tuned DistilBERT model trained on IMDB reviews. Enter any text to analyze its sentiment.",
    examples=[
        ["This movie was absolutely fantastic! One of the best films I have ever seen."],
        ["I hated every minute of this film. Complete waste of time."],
        ["It was okay, nothing special but not terrible either."]
    ]
)

demo.launch()
