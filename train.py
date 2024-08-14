import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import tiktoken
from model import GreesyGuard
from tqdm import tqdm
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, texts, labels, label_to_id, tokenizer, max_length=2048):
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
        tokens = tokens + [0] * (self.max_length - len(tokens))  # Padding
        return torch.tensor(tokens), torch.tensor(label, dtype=torch.long)

def train(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            category_scores = model(inputs)
            loss = criterion(category_scores, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
        
   









        


    return model

def main():
    # Load dataset
    dataset = load_dataset("OnlyCheeini/Greesyguard-2.5-mini")  # Replace with actual dataset
    train_data = dataset['train']
   

    # Get categories from dataset
    all_categories = train_data['category'] 
    category_counts = Counter(all_categories)
    categories = list(category_counts.keys())
    label_to_id = {label: id for id, label in enumerate(categories)}
    id_to_label = {id: label for label, id in label_to_id.items()}

    print(f"Categories: {categories}")
    print(f"Category counts: {category_counts}")

    # Tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Prepare datasets
    train_dataset = TextDataset(train_data['text'], train_data['category'], label_to_id, tokenizer)
    #val_dataset = TextDataset(val_data['text'], val_data['category'], label_to_id, tokenizer)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=32)

    # Model initialization
    model = GreesyGuard(
        vocab_size=150000,
        embed_dim=256,
        hidden_dim=128,
        num_categories=len(categories)
    )

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    # Train the model
    model = train(model, train_loader, criterion, optimizer, num_epochs=5, device=device)

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'categories': categories,
        'label_to_id': label_to_id,
        'id_to_label': id_to_label
    }, 'greesyguard.pth')

if __name__ == "__main__":
    main()
