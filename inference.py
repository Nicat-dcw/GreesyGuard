import torch
from pathlib import Path
from model import GreesyGPT, generate_moderation, ReasoningMode, OutputFormat, DEVICE

# Initialize model
model = GreesyGPT()

# Load trained weights if they exist
weights_path = Path(__file__).parent / "greesy_gpt.pt"
if weights_path.exists():
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print(f"Loaded weights from {weights_path}")
else:
    print("No trained weights found, using fresh initialization.")

model.to(DEVICE)

# Run a 'MEDIUM' moderation check
result = generate_moderation(
    model, 
    prompt="You're so stupid, nobody likes you.",
    mode=ReasoningMode.MEDIUM,
    output_format=OutputFormat.JSON
)

# Access structured data
print(result["verdict_fmt"]["verdict"])  # e.g., "HARASSMENT"
print(result["thinking"])                # e.g., "The user is using targeted insults..."
