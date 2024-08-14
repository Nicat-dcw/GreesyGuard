import torch
import tiktoken
from model import GreesyGuard
from typing import Dict, List
import json

class TextAnalyzer:
    def __init__(self, model_path: str):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model and metadata
        checkpoint = torch.load(model_path, map_location=self.device)
        self.categories = checkpoint['categories']
        self.label_to_id = checkpoint['label_to_id']
        self.id_to_label = checkpoint['id_to_label']
        
        self.model = GreesyGuard(
            vocab_size=150000, 
            embed_dim=256, 
            hidden_dim=128, 
            num_categories=len(self.categories)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def analyze_text(self, text: str) -> Dict:
        tokens = torch.tensor([self.tokenizer.encode(text)[:256]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            category_scores = self.model(tokens)
            probabilities = torch.softmax(category_scores, dim=1)

        results = {
            "flagged": False,
            "categories": {category: False for category in self.categories},
            "category_scores": {category: 0.0 for category in self.categories}
        }

        for i, category in enumerate(self.categories):
            score = probabilities[0][i].item()
            results["category_scores"][category] = score
            if score > 0.5:  # Threshold for flagging
                results["categories"][category] = True
                results["flagged"] = True

        return results

    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        return [self.analyze_text(text) for text in texts]

def main():
    analyzer = TextAnalyzer('greesyguard.pth')

    test_texts = [
        "You're all idiots and should die!",
        "I'm feeling really depressed and don't want to live anymore.",
        "Let's meet up for coffee and chat about the new project.",
        "I'll beat you up if you don't give me your money right now!",
        "This movie has a lot of nudity and explicit scenes.",
    ]

    for text in test_texts:
        results = analyzer.analyze_text(text)
        print(f"\nText: '{text}'")
        print(json.dumps(results, indent=2))
        print("---")

if __name__ == "__main__":
    main()
