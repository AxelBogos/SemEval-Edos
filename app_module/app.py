from pathlib import Path

import matplotlib.pyplot as plt
import shap
import streamlit as st
import torch

from app_module import helpers


# Define a helper function to load models
def load_model(model_path):
    model = torch.load(model_path)
    return model


# Load models
models_path = Path("saved_models")
model_a = load_model(models_path / "task_a" / "model.pt")
model_b = load_model(models_path / "task_b" / "model.pt")
model_c = load_model(models_path / "task_c" / "model.pt")

models = {"Task A": model_a, "Task B": model_b, "Task C": model_c}

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

# Button to run the models
if st.button("Submit"):
    predictions = {}
    with st.spinner("Classifying..."):
        for model_name, model in models.items():
            prediction = model.predict(statement)  # Replace with your models' predict method
            predictions[model_name] = prediction

    st.write("Predicted classes:")
    for model_name, prediction in predictions.items():
        st.write(f"{model_name}: {prediction}")

    # Explanation of the predictions
    model_to_explain = st.selectbox("Select a model to get an explanation", list(models.keys()))
    explanation_method = st.selectbox(
        "Select an explanation method", ["SHAP Values", "Attention Scores"]
    )

    if st.button("Explain"):
        with st.spinner("Calculating explanation..."):
            model = models[model_to_explain]

            if explanation_method == "SHAP Values":
                explainer = shap.DeepExplainer(
                    model, statement
                )  # Replace with appropriate SHAP implementation
                shap_values = explainer.shap_values(statement)

                # Visualize SHAP values
                plt.figure(figsize=(10, 5))
                plt.bar(range(len(shap_values)), shap_values)
                plt.xlabel("Feature Index")
                plt.ylabel("SHAP Value")
                plt.title(f"SHAP Values for {model_to_explain}")
                st.pyplot(plt.gcf())

            elif explanation_method == "Attention Scores":
                attention_scores = helpers.get_attention_scores(
                    model, statement
                )  # Implement your own attention score function

                # Visualize attention scores
                plt.figure(figsize=(10, 5))
                plt.bar(range(len(attention_scores)), attention_scores)
                plt.xlabel("Token Index")
                plt.ylabel("Attention Score")
                plt.title(f"Attention Scores for {model_to_explain}")
                st.pyplot(plt.gcf())
