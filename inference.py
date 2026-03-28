from model import GreesyGPT, generate_moderation, ReasoningMode, OutputFormat

# Initialize model (ensure you have trained weights or initialize fresh)
model = GreesyGPT()

# Run a 'Deep' moderation check
result = generate_moderation(
    model, 
    prompt="You're so stupid, nobody likes you.",
    mode=ReasoningMode.DEEP,
    output_format=OutputFormat.JSON
)

# Access structured data
print(result["verdict_fmt"]["verdict"])  # e.g., "HARASSMENT"
print(result["thinking"])                # e.g., "The user is using targeted insults..."