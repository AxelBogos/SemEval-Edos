from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import shap
import torch
from dash import dcc, html
from dash.dependencies import Input, Output, State
from transformers import AutoTokenizer

from src.models.transformer_module import TransformerModule
from src.utils import defines

# Load models
model_a = TransformerModule.load_from_checkpoint(
    defines.SAVED_MODEL_DIR / "task_a.ckpt", map_location=torch.device("cpu")
)
model_b = TransformerModule.load_from_checkpoint(
    defines.SAVED_MODEL_DIR / "task_b.ckpt", map_location=torch.device("cpu")
)
model_c = TransformerModule.load_from_checkpoint(
    defines.SAVED_MODEL_DIR / "task_c.ckpt", map_location=torch.device("cpu")
)

models = {"Task A": model_a, "Task B": model_b, "Task C": model_c}

# Provide examples of sexist statements
example_statements = ["Example 1", "Example 2", "Example 3"]


# Get attention scores function
def get_attention_scores(model, statement):
    model.eval()  # Set model to evaluation mode

    # Tokenize the input statement
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    inputs = tokenizer(statement, return_tensors="pt")

    # Run the model to get attention scores
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Extract and average attention scores across all layers and heads
    attentions = outputs.attentions  # List of attention matrices for each layer
    attentions = torch.stack(attentions).squeeze(1)  # Stack all attention matrices
    avg_attention = attentions.mean(dim=(0, 1)).cpu().numpy()  # Calculate mean attention scores

    return avg_attention


# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row([dbc.Col(html.H1("Sexist Statement Classifier"), className="mb-2")]),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Form(
                        [
                            html.Label("Example statements"),
                            dcc.Dropdown(
                                id="example-statements",
                                options=[
                                    {"label": example, "value": example}
                                    for example in example_statements
                                ],
                                value=None,
                            ),
                            html.Label("Or enter your own statement"),
                            dbc.Input(id="user-input", type="text"),
                            dbc.Button(
                                "Submit", id="submit-button", color="primary", className="mt-2"
                            ),
                        ]
                    ),
                    className="mb-4",
                )
            ]
        ),
        dbc.Row([dbc.Col(html.Div(id="predictions"), className="mb-4")]),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Form(
                        [
                            html.Label("Select a model to get an explanation"),
                            dcc.Dropdown(
                                id="model-selection",
                                options=[
                                    {"label": model_name, "value": model_name}
                                    for model_name in models.keys()
                                ],
                                value=None,
                            ),
                            html.Label("Select an explanation method"),
                            dcc.Dropdown(
                                id="explanation-method",
                                options=[
                                    {"label": method, "value": method}
                                    for method in ["SHAP Values", "Attention Scores"]
                                ],
                                value=None,
                            ),
                            dbc.Button(
                                "Explain", id="explain-button", color="primary", className="mt-2"
                            ),
                        ]
                    ),
                    className="mb-4",
                )
            ]
        ),
        dbc.Row([dbc.Col(html.Div(id="explanation"), className="mb-4")]),
    ]
)


@app.callback(
    Output("predictions", "children"),
    Input("submit-button", "n_clicks"),
    State("example-statements", "value"),
    State("user-input", "value"),
)
def update_predictions(n_clicks, selected_example, user_input):
    if n_clicks:
        statement = user_input if user_input else selected_example
        predictions = {}
        for model_name, model in models.items():
            prediction = model(statement)  # Replace with your models' predict method
            predictions[model_name] = prediction

        return [
            html.P(f"{model_name}: {prediction}") for model_name, prediction in predictions.items()
        ]

    return None


@app.callback(
    Output("explanation", "children"),
    Input("explain-button", "n_clicks"),
    State("model-selection", "value"),
    State("explanation-method", "value"),
    State("user-input", "value"),
    State("example-statements", "value"),
)
def update_explanation(
    n_clicks, model_to_explain, explanation_method, user_input, selected_example
):
    if n_clicks:
        statement = user_input if user_input else selected_example
        model = models[model_to_explain]

        if explanation_method == "SHAP Values":
            explainer = shap.DeepExplainer(
                model, statement
            )  # Replace with appropriate SHAP implementation
            shap_values = explainer.shap_values(statement)

            # Visualize SHAP values
            shap_plot = go.Figure(go.Bar(x=list(range(len(shap_values))), y=shap_values))
            shap_plot.update_layout(
                title=f"SHAP Values for {model_to_explain}",
                xaxis_title="Feature Index",
                yaxis_title="SHAP Value",
            )
            return dcc.Graph(figure=shap_plot)

        elif explanation_method == "Attention Scores":
            attention_scores = get_attention_scores(
                model, statement
            )  # Use the get_attention_scores function

            # Visualize attention scores
            attention_plot = go.Figure(
                go.Bar(x=list(range(len(attention_scores))), y=attention_scores)
            )
            attention_plot.update_layout(
                title=f"Attention Scores for {model_to_explain}",
                xaxis_title="Token Index",
                yaxis_title="Attention Score",
            )
            return dcc.Graph(figure=attention_plot)

    return None


if __name__ == "__main__":
    app.run_server(debug=True)
