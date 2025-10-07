import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path

# -----------------------------
# Project paths
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent  # project root relative to this script
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
NER_DATA_PATH = DATA_DIR / "ner_data.json"
NER_MODEL_PATH = MODELS_DIR / "ner_model"

# Create models directory if it does not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load NER data
# -----------------------------
with open(NER_DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)  # list of dicts with 'text' and 'entities'

# -----------------------------
# Custom Dataset for NER
# -----------------------------
class NERDataset(Dataset):
    # PyTorch dataset for token classification
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {"O": 0, "ANIMAL": 1}  # extend if more labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        labels = [0] * self.max_len
        for start, end, label in item["entities"]:
            for i in range(int(start), int(end)):
                if i < self.max_len:
                    labels[i] = self.label_map[label]

        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(labels)
        return enc

# -----------------------------
# Tokenizer and DataLoader
# -----------------------------
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = NERDataset(data, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------------
# Model setup
# -----------------------------
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# -----------------------------
# Training loop with token-level accuracy
# -----------------------------
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0

    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Token-level accuracy
        preds = outputs.logits.argmax(dim=-1)
        mask = attention_mask.bool()
        correct_tokens += (preds[mask] == labels[mask]).sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct_tokens / total_tokens
    print(f"Epoch {epoch+1}/{num_epochs}, Loss={avg_loss:.4f}, Token Accuracy={accuracy:.4f}")

# -----------------------------
# Save trained model and tokenizer
# -----------------------------
model.save_pretrained(NER_MODEL_PATH)
tokenizer.save_pretrained(NER_MODEL_PATH)
print(f"✅ NER model saved to: {NER_MODEL_PATH}")
