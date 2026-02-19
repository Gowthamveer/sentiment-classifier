
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

# Load dataset
dataset = load_dataset("imdb")

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=1000)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# Subsets
small_train = tokenized_dataset["train"].shuffle(seed=42).select(range(3000))
small_test  = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

train_loader = DataLoader(small_train, batch_size=16, shuffle=True)
test_loader  = DataLoader(small_test,  batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in loop:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total * 100
    print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

# Save model
model.save_pretrained("./models")
tokenizer.save_pretrained("./models")
print("Model saved!")
