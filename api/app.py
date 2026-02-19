
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = FastAPI(title="Sentiment Classifier API", version="1.0")

MODEL_PATH = "/content/drive/MyDrive/sentiment-classifier/models"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

@app.get("/")
def home():
    return {"message": "Sentiment Classifier API is running!"}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: TextRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    sentiment = "positive" if pred == 1 else "negative"
    return SentimentResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=round(confidence * 100, 2)
    )

@app.get("/health")
def health():
    return {"status": "healthy"}
