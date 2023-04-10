import torch
from transformers import AutoTokenizer


# Define a helper function to load models
def load_model(model_path):
    model = torch.load(model_path)
    return model


def get_attention_scores(model, statement):
    model.eval()  # Set model to evaluation mode

    # Tokenize the input statement
    tokenizer = AutoTokenizer.from_pretrained("distil-roberta-base")
    inputs = tokenizer(statement, return_tensors="pt")

    # Run the model to get attention scores
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Extract and average attention scores across all layers and heads
    attentions = outputs.attentions  # List of attention matrices for each layer
    attentions = torch.stack(attentions).squeeze(1)  # Stack all attention matrices
    avg_attention = attentions.mean(dim=(0, 1)).cpu().numpy()  # Calculate mean attention scores

    return avg_attention
