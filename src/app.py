from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import shap
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoTokenizer

from models.transformer_module import TransformerModule
from utils import defines

saved_models_dir = Path("saved_models").resolve()


def main():
    # Load models
    models = load_models()
    label_dicts = get_label_dicts()
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    # Streamlit interface
    st.title("Sexist Statement Classifier")
    st.write("Select or input a statement to classify:")

    # Provide examples of sexist statements
    example_statements = ["Example 1", "Example 2", "Example 3"]
    selected_example = st.selectbox("Example statements", example_statements)

    # Allow user to input a statement
    user_input = st.text_input("Or enter your own statement")

    # Choose statement to classify
    statement = user_input if user_input else selected_example
    logits = {}
    predictions = {}
    if st.button("Submit"):
        inputs = tokenizer(statement, return_tensors="pt")
        with st.spinner("Classifying..."):
            for model_name, model in models.items():
                outputs = model(**inputs)
                logits[model_name] = outputs.logits
                predictions[model_name] = outputs.logits.argmax().item()

        st.write("Predicted classes:")
        for model_name, prediction in predictions.items():
            st.write(f"{model_name}: {label_dicts[model_name][prediction]}")

        # Explanation of the predictions
        st.write("Explanation of the predictions:")
        for model_name, logit in logits.items():
            # Extract the predicted class probabilities
            pred_prob = torch.softmax(logit, dim=1).detach().cpu().numpy()[0]

            # Initialize the explainer
            explainer = shap.Explainer(models[model_name], tokenizer)

            # Generate the Shapley values for the input text
            input_ids = inputs["input_ids"].unsqueeze(0)
            shap_values = explainer(input_ids)

            # Plot the Shapley values
            st.write(f"{model_name} explanation:")
            shap.plots.text(shap_values[0], feature_names=tokenizer.decode(inputs[0]))


@st.cache_resource
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
        num_target_class=11,
        num_epochs=9,
        lr=5e-6,
        len_train_loader=100,
        n_warmup_steps=0,
    )
    return args_task_a, args_task_b, args_task_c


@st.cache_resource
def load_models():
    args_task_a, args_task_b, args_task_c = get_task_namespace()
    model_a = TransformerModule.load_from_checkpoint(
        defines.SAVED_MODEL_DIR / "task_a.ckpt",
        map_location=torch.device("cpu"),
        args=args_task_a,
        optimizer=torch.optim.AdamW,
    )
    model_b = TransformerModule.load_from_checkpoint(
        defines.SAVED_MODEL_DIR / "task_b.ckpt",
        map_location=torch.device("cpu"),
        args=args_task_b,
        optimizer=torch.optim.AdamW,
    )
    model_c = TransformerModule.load_from_checkpoint(
        defines.SAVED_MODEL_DIR / "task_c.ckpt",
        map_location=torch.device("cpu"),
        args=args_task_c,
        optimizer=torch.optim.AdamW,
    )
    return {"Task A": model_a, "Task B": model_b, "Task C": model_c}


@st.cache_resource
def get_label_dicts() -> dict[str, Union[dict[int, str], dict[Union[int, Any], Union[str, Any]]]]:
    task_a = {0: " Not Sexist", 1: "Sexist"}
    task_b = {
        0: "Threats, plans to harm and incitement",
        1: "Derogation",
        2: "Animosity",
        3: "Prejudiced discussions",
    }
    task_c = {
        0: "Threats of harm",
        1: "Incitement and encouragement of harm",
        2: "Descriptive attacks",
        3: "Aggressive and emotive attacks",
        4: "Dehumanising attacks & overt sexual objectification",
        5: "Casual use of gendered slurs, profanities, and insults",
        6: "Immutable gender differences and gender stereotypes",
        7: "Backhanded gendered compliments",
        8: "Condescending explanations or unwelcome advice",
        9: "Supporting mistreatment of individual women",
        10: "Supporting systemic discrimination against women as a group",
    }
    return {"Task A": task_a, "Task B": task_b, "Task C": task_c}


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


if __name__ == "__main__":
    main()
