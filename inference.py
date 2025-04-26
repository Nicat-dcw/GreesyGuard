import torch
import tiktoken
from model import GreesyGuard
from typing import Dict, List
import json

class TextAnalyzer:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold

        # Load model and metadata
        checkpoint = torch.load(model_path, map_location=self.device)

        # Print raw categories for debugging
        print("Loaded categories from checkpoint:", checkpoint.get('categories'))

        # Use raw categories as is (without cleaning or filtering)
        self.categories = checkpoint.get('categories', [])
        self.label_to_id = checkpoint.get('label_to_id', {})
        self.id_to_label = checkpoint.get('id_to_label', {})

        self.model = GreesyGuard(
            vocab_size = 100_000,
            embed_dim  = 128,
            hidden_dim = 32,
# classifier input dim = hidden_dim * 2 (because it’s a    Bi-LSTM) = 64
            num_categories=len(self.categories)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def analyze_text(self, text: str) -> Dict:
        tokens = torch.tensor([self.tokenizer.encode(text)[:256]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            category_scores = self.model(tokens)
            probabilities = torch.softmax(category_scores, dim=1)

        results = {
            "text": text,
            "flagged": False,
            "categories": {category: False for category in self.categories},
            "category_scores": {category: 0.0 for category in self.categories}
        }

        for i, category in enumerate(self.categories):
            score = probabilities[0][i].item()
            results["category_scores"][category] = score
            if score > self.threshold:
                results["categories"][category] = True
                results["flagged"] = True

        # Print top prediction for inspection
        top_score, top_index = torch.max(probabilities, dim=1)
        print(f"\nTop category: '{self.categories[top_index.item()]}' with score: {top_score.item():.4f}")

        return results

    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        return [self.analyze_text(text) for text in texts]

def main():
    analyzer = TextAnalyzer('greesyguard_final.pth', threshold=0.3)

    test_texts = [
        "Watch your back, because you're on my list.",
        "hi"
    ]

    with open("results.jsonl", "w") as f:
        for text in test_texts:
            results = analyzer.analyze_text(text)
            print(f"\nText: '{text}'")
            print(json.dumps(results, indent=2))
            print("---")
            f.write(json.dumps(results) + "\n")

if __name__ == "__main__":
    main()
