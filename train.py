import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import tiktoken
from model import GreesyGuard
# Load the dataset
dataset = load_dataset("badmatr11x/hate-offensive-speech")


# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Custom dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = self.tokenizer.encode(item['tweet'])[:self.max_length]
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

# Prepare datasets
train_dataset = HateSpeechDataset(dataset['train'], tokenizer)
test_dataset = HateSpeechDataset(dataset['test'], tokenizer)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model
conf = { "vocab_size": 150000, "embed_dim":128, "hidden_dim":64, "output_dim":3}
model = GreesyGuard(vocab_size=150000, embed_dim=128, hidden_dim=64, output_dim=3)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Params:{total_params}')
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss/len(train_loader):.4f}')
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    print('---')

# Save the model
torch.save(model.state_dict(), 'model.bin')
model.push_to_hub("greesyguard")
print("Model saved successfully!")
