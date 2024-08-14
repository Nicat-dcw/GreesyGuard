mport torch
from torch import nn

class GreesyGuard(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_categories):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_categories)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        category_scores = self.classifier(pooled)
        return category_scores
