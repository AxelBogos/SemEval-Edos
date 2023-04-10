from argparse import Namespace

import torch
from transformers import AutoTokenizer

from src.models.transformer_module import TransformerModule
from src.utils import defines


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


def get_task_namespace():
    args_task_a = Namespace(
        model="distilroberta-base",
        num_target_class=2,
        num_epochs=9,
        lr=5e-6,
        len_train_loader=100,
        n_warmup_steps=0,
    )
    args_task_b = Namespace(
        model="distilroberta-base",
        num_target_class=4,
        num_epochs=9,
        lr=5e-6,
        len_train_loader=100,
        n_warmup_steps=0,
    )
    args_task_c = Namespace(
        model="distilroberta-base",
        num_target_class=10,
        num_epochs=9,
        lr=5e-6,
        len_train_loader=100,
        n_warmup_steps=0,
    )
    return args_task_a, args_task_b, args_task_c


def load_models():
    args_task_a, args_task_b, args_task_c = get_task_namespace()
    model_a = TransformerModule.load_from_checkpoint(
        defines.SAVED_MODEL_DIR / "task_a.ckpt",
        map_location=torch.device("cpu"),
        args=args_task_a,
        optimizer=torch.optim.Adamw,
    )
    model_b = TransformerModule.load_from_checkpoint(
        defines.SAVED_MODEL_DIR / "task_b.ckpt",
        map_location=torch.device("cpu"),
        args=args_task_b,
        optimizer=torch.optim.Adamw,
    )
    model_c = TransformerModule.load_from_checkpoint(
        defines.SAVED_MODEL_DIR / "task_a.ckpt",
        map_location=torch.device("cpu"),
        args=args_task_c,
        optimizer=torch.optim.Adamw,
    )
    return {"Task A": model_a, "Task B": model_b, "Task C": model_c}
