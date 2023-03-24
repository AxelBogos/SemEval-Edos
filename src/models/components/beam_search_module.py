import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class BeamSearchModule(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).base_model
        self.TaskA_clf = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).classifier
        self.TaskB_clf = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=5
        ).classifier
        self.TaskC_clf = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=12
        ).classifier
        self.task_children = self._get_task_children

    def forward(self, input_ids, attention_mask, labels=None):
        candidate_labels = [-1, -1, -1]
        encoded_output = self.encoder(input_ids, attention_mask=attention_mask, labels=labels)
        task_a_output = self.TaskA_clf(encoded_output.last_hidden_state)
        task_b_output = self.TaskB_clf(encoded_output.last_hidden_state)
        task_c_output = self.TaskC_clf(encoded_output.last_hidden_state)

        taskA_label = torch.argmax(task_a_output, dim=1)

    @property
    def _get_valid_children(self):
        return [
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 2][1, 2, 1],
            [1, 2, 2],
            [1, 2, 3],
            [1, 3, 1],
            [1, 3, 2],
            [1, 3, 3],
            [1, 3, 4],
            [1, 4, 1],
            [1, 4, 2],
        ]

    @property
    def _get_task_children(self):
        task_children = {}
        return task_children

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_module(module):
        for param in module.parameters():
            param.requires_grad = True
