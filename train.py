import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import tiktoken
from model import GreesyGuard
from tqdm import tqdm
from collections import Counter
import gc
import os
import csv
import requests
import time
from pathlib import Path

class TextDataset(Dataset):
    def __init__(self, texts, labels, label_to_id, tokenizer, max_length=1024):
        self.texts = texts
        self.labels = labels
        self.label_to_id = label_to_id
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label_to_id[self.labels[idx]]
        tokens = self.tokenizer.encode(text)[:self.max_length]
        return torch.tensor(tokens), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    """Custom collate function for dynamic padding"""
    tokens, labels = zip(*batch)
    max_len = max(len(seq) for seq in tokens)
    padded_tokens = [seq.tolist() + [0] * (max_len - len(seq)) for seq in tokens]
    return torch.tensor(padded_tokens), torch.stack(labels)

def download_dataset(url, data_dir='./data'):
    Path(data_dir).mkdir(exist_ok=True)
    local_file = os.path.join(data_dir, 'dataset.csv')
    if os.path.exists(local_file):
        print(f"Using existing dataset at {local_file}")
        return local_file
    print(f"Downloading dataset from {url}")
    response = requests.get(url)
    response.raise_for_status()
    with open(local_file, 'wb') as f:
        f.write(response.content)
    print(f"Dataset saved to {local_file}")
    return local_file

def load_csv_dataset(file_path):
    print(f"Loading CSV dataset from {file_path}")
    texts, categories, models = [], [], []
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(1024)
                f.seek(0)
                delimiter = ',' if ',' in sample else (';' if ';' in sample else ('\t' if '\t' in sample else ','))
                has_header = 'text' in sample.lower() and 'category' in sample.lower()
                reader = csv.reader(f, delimiter=delimiter)
                if has_header:
                    header = next(reader)
                    text_idx = next((i for i, col in enumerate(header) if 'text' in col.lower()), 0)
                    category_idx = next((i for i, col in enumerate(header) if 'category' in col.lower()), 1)
                    model_idx = next((i for i, col in enumerate(header) if 'model' in col.lower()), 2)
                else:
                    text_idx, category_idx, model_idx = 0, 1, 2
                for row in reader:
                    if len(row) > category_idx:
                        texts.append(row[text_idx])
                        categories.append(row[category_idx])
                        if len(row) > model_idx:
                            models.append(row[model_idx])
                break
        except Exception as e:
            print(f"Failed with encoding {encoding}: {e}")
            if encoding == encodings[-1]:
                raise
    print(f"Loaded {len(texts)} examples")
    return texts, categories, models


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=3):
    best_val_acc, epochs_no_improve = 0, 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        accumulation_steps = 4
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)
            scores = model(inputs)
            loss = criterion(scores, labels) / accumulation_steps
            loss.backward()
            if (i+1) % accumulation_steps == 0 or (i+1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * accumulation_steps
            del inputs, labels, scores, loss
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Time: {time.time()-start_time:.2f}s")
        model.eval()
        correct = total = val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                scores = model(inputs)
                loss = criterion(scores, labels)
                val_loss += loss.item()
                _, preds = torch.max(scores, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                del inputs, labels, scores, loss
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss/len(val_loader):.4f}")
        scheduler.step(val_loss/len(val_loader))
        if val_acc > best_val_acc:
            best_val_acc, epochs_no_improve = val_acc, 0
            torch.save({'model_state_dict': model.state_dict(),'val_accuracy': val_acc,'epoch': epoch}, 'greesyguard_best.pth')
            print(f"Saved best model with accuracy: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    return model


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    torch.set_num_threads(4)
    dataset_path = None
    for path in ['dataset.csv']:
        if os.path.exists(path):
            dataset_path = path
            print(f"Found dataset at {path}")
            break
    if not dataset_path and os.path.exists('dataset_url.txt'):
        with open('dataset_url.txt','r') as f:
            url = f.read().strip()
        dataset_path = download_dataset(url)
    if not dataset_path:
        user_path = input("Path to CSV file (e.g., 'data/mydata.csv'): ")
        if os.path.exists(user_path): dataset_path = user_path
        else:
            print(f"Error: File not found at {user_path}")
            return
    texts, categories, models = load_csv_dataset(dataset_path)
    category_counts = Counter(categories)
    unique_categories = list(category_counts.keys())
    label_to_id = {l:i for i,l in enumerate(unique_categories)}
    print(f"Categories: {unique_categories}")
    print(f"Category counts: {category_counts}")
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        print("Using cl100k_base tokenizer")
    except:
        class SimpleTokenizer:
            def encode(self, text): return [ord(c)%100000 for c in text]
        tokenizer = SimpleTokenizer()
        print("Tiktoken failed, using SimpleTokenizer")

    full_dataset = TextDataset(texts, categories, label_to_id, tokenizer)
    # Train on just 50% of the data
    total_samples = len(full_dataset)
    train_size = int(0.5 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Training on {train_size} samples ({100*train_size/total_samples:.1f}%), validating on {val_size} samples")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn, num_workers=0)

    model = GreesyGuard(vocab_size=100000, embed_dim=128, hidden_dim=32, num_categories=len(unique_categories))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    if len(set(category_counts.values())) > 1:
        counts = [category_counts[c] for c in unique_categories]
        weights = 1.0/torch.tensor(counts, dtype=torch.float)
        weights = weights/weights.sum()*len(unique_categories)
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using weighted loss")
    else:
        criterion = nn.CrossEntropyLoss()

    try:
        trained = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
        torch.save({'model_state_dict': trained.state_dict(),'categories': unique_categories,'label_to_id': label_to_id}, 'greesyguard_final.pth')
        print("Training completed!")
    except KeyboardInterrupt:
        print("Training interrupted, saving model...")
        torch.save({'model_state_dict': model.state_dict(),'categories': unique_categories,'label_to_id': label_to_id}, 'greesyguard_interrupted.pth')
        print("Model saved as greesyguard_interrupted.pth")

if __name__ == "__main__":
    main()
