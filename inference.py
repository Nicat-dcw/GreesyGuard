import torch
import tiktoken
from model import GreesyGuard 
import time
from pydantic import BaseModel
from fastapi import FastAPI

# Load the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GreesyGuard(vocab_size=150000, embed_dim=128, hidden_dim=64, output_dim=3)
model.load_state_dict(torch.load('model.bin', map_location=device))
model.to(device)
model.eval()


def classify_text(text):
    tokens = torch.tensor([tokenizer.encode(text)[:128]], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(tokens)
        _, predicted = torch.max(outputs.data, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    label_map = {0: "hate-speech", 1: "offensive-speech", 2: "neither"}
    return label_map[predicted.item()], probabilities[0][predicted].item()

app = FastAPI(title="OpenAI-compatible API")
class Moderations(BaseModel):
    model: str = "greesyguard-2-1m",
    input: str = "",
    beta: bool
@app.post("/v1/moderations")
async def mod(request: Moderations):
    
    if request.input and request.input == 'null':
      resp_content = "Enter a message:" 
    else:
      resp_content = "Enter input!"
      classification, confidence = classify_text(request.input)
    print(request.input)
    return {
        "id": "1337",
        "created": time.time(),
        "model": request.model,
        "results": [{
            "categories": classification,
            "flagged": classification in ['hate-speech', 'offensive-speech'],
            "score": confidence
        }]
    }

# Test the classifier
test_texts = [
    "where's his other half??? i want to indulge myself with a lot of yunho esp him dancing keep you head up today! ",
    "I love spending time with my family.",
    "Die, you scum!",
    "The weather is nice today."
]

for text in test_texts:
    classification, confidence = classify_text(text)
    print(f"Text: '{text}'")
    print(f"Classification: {classification}")
    print(f"Confidence Score: {confidence:.4f}")
    print('---')
import uvicorn 

if __name__ == "__main__":
    uvicorn.run("main.app:app", host="0.0.0.0", reload=True)