import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tiktoken


class TiktokenTokenizer:
    def __init__(self, model_name="p50k_base"):
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.pad_token_id = 0  # padding

    def encode_plus(self, text, max_length, padding, truncation):
        tokens = self.tokenizer.encode(text)
        if truncation:
            tokens = tokens[:max_length]
        if padding == 'max_length':
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
        attention_mask = [1] * len(tokens)
        return {
            'input_ids': torch.tensor(tokens),
            'attention_mask': torch.tensor(attention_mask)
        }

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def __len__(self):
        return self.tokenizer.n_vocab

class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

class GreesyGuard(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        pooled = (embedded * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).unsqueeze(-1)
        return self.fc(pooled)


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = nn.functional.cross_entropy(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = nn.functional.cross_entropy(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
    return total_loss / len(test_loader), correct / len(test_loader.dataset)


if __name__ == '__main__':
    # Load data
    data = load_data('data.json')
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=3407)

    tokenizer = TiktokenTokenizer()
    max_length = 128
    vocab_size = len(tokenizer)
    embed_dim = 256
    num_classes = 2
    model = GreesyGuard(vocab_size, embed_dim, num_classes)

    train_dataset = TextClassificationDataset(train_data, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_data, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Set up optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss = train(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f'Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

    # Save the model
    torch.save(model.state_dict(), './greesyguard.safetensors')

    def classify_text_with_score(text):
        model.eval()
        encoding = tokenizer.encode_plus(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True
        )
        input_ids = encoding['input_ids'].unsqueeze(0).to(device)
        attention_mask = encoding['attention_mask'].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        # Get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0][pred_class].item()
        
        class_names = ['Not Blocked', 'Blocked']
        return class_names[pred_class], pred_prob

    
    sample_texts = [
        "Nude",
        "beautiful landscape",
        "explicit content",
        "family friendly"
    ]
    t = "Nude"
    print("Classification results:")
    prediction, score = classify_text_with_score(t)
    print(f"Text: {t}""\nPrediction:{prediction}"f"Score:{score}")
    