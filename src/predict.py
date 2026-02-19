
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_PATH = "./models"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
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
    print(f"Text       : {text}")
    print(f"Sentiment  : {label}")
    print(f"Confidence : {confidence:.2f}%")

if __name__ == "__main__":
    predict_sentiment("This movie was absolutely amazing!")
    predict_sentiment("I hated every minute of this film.")
