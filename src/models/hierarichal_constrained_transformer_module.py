import lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassF1Score
from transformers import AutoModelForSequenceClassification, get_scheduler


class HierarchicalTransformerModule(pl.LightningModule):
    def __init__(
        self,
        model: str,
        learning_rate: float,
        classifier_a: nn.Module,
        classifier_b: nn.Module,
        classifier_c: nn.Module,
        task: str,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        freeze_classification_heads: bool = False,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.optimizer = optimizer
        self.model = model
        self.learning_rate = learning_rate
        self.task = task
        self.feature_extractor = AutoModelForSequenceClassification.from_pretrained(
            model
        ).base_model
        self.classifier_a = classifier_a
        self.classifier_b = classifier_b
        self.classifier_c = classifier_c
        self.freeze_module(self.feature_extractor)
        if freeze_classification_heads:
            self.freeze_module(self.classifier_a)
            self.freeze_module(self.classifier_b)
            self.freeze_module(self.classifier_c)

        self.criterion = nn.CrossEntropyLoss()

        if self.task == "a":
            self.num_target_class = 2
        elif self.task == "b":
            self.num_target_class = 4
        elif self.task == "c":
            self.num_target_class = 11

        self.train_f1 = MulticlassF1Score(num_classes=self.num_target_class, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=self.num_target_class, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=self.num_target_class, average="macro")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_f1_best = MaxMetric()
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        features = self.feature_extractor(input_ids, attention_mask).last_hidden_state
        logits_a = self.classifier_a(features)
        logits_b = self.classifier_b(features)
        logits_c = self.classifier_c(features)
        return logits_a, logits_b, logits_c

    def apply_constraints(self, logits):
        logits_a, logits_b, logits_c = logits

        # Apply the hierarchical constraints for level A and B
        preds_a = torch.argmax(logits_a, dim=1)

        allowed_indices_b = torch.zeros_like(logits_b, dtype=torch.bool)
        allowed_indices_b[:, 0][preds_a == 0] = True
        allowed_indices_b[:, 1:][preds_a == 1] = True

        logits_b[~allowed_indices_b] = float("-inf")
        preds_b = torch.argmax(logits_b, dim=1)

        # Apply the hierarchical constraints for level B and C
        allowed_indices_c = torch.zeros_like(logits_c, dtype=torch.bool)
        allowed_indices_c[:, 0][preds_b == 0] = True
        allowed_indices_c[:, 1:3][preds_b == 1] = True
        allowed_indices_c[:, 3:6][preds_b == 2] = True
        allowed_indices_c[:, 6:10][preds_b == 3] = True
        allowed_indices_c[:, 10:12][preds_b == 4] = True

        logits_c[~allowed_indices_c] = float("-inf")
        preds_c = torch.argmax(logits_c, dim=1)

        if self.task == "a":
            return preds_a
        elif self.task == "b":
            return preds_b
        elif self.task == "c":
            return preds_c
        else:
            raise ValueError("Invalid task")

    def model_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        full_logits = self(input_ids, attention_mask)
        if self.task == "a":
            logits = full_logits[0]
        elif self.task == "b":
            logits = full_logits[1]
        elif self.task == "c":
            logits = full_logits[2]
        loss = self.criterion(logits, labels)
        preds = self.apply_constraints(full_logits)

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.model_step(batch)

        # Log
        self.train_loss(loss)
        self.train_f1(preds, labels)
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.model_step(batch)

        # Log
        self.val_loss(loss)
        self.val_f1(preds, labels)
        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.model_step(batch)

        # Log
        self.test_loss(loss)
        self.test_f1(preds, labels)
        self.log("test/loss", self.test_loss, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        self.val_f1_best.update(f1)
        self.log("val/f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.task == "a":
            optimizer = self.optimizer(self.classifier_a.parameters(), lr=self.learning_rate)
        elif self.task == "b":
            optimizer = self.optimizer(self.classifier_b.parameters(), lr=self.learning_rate)
        elif self.task == "c":
            optimizer = self.optimizer(self.classifier_c.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_module(module):
        for param in module.parameters():
            param.requires_grad = True
