import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassF1Score
from transformers import AutoModelForSequenceClassification, get_scheduler

from src.models.transformer_module import TransformerModule


class WrapperTransformerModule(pl.LightningModule):
    def __init__(
        self,
        args_task_a: argparse.Namespace,
        args_task_b: argparse.Namespace,
        args_task_c: argparse.Namespace,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.args_task_a = args_task_a
        self.args_task_b = args_task_b
        self.args_task_c = args_task_c
        (
            self.feature_extractor,
            self.classifier_a,
            self.classifier_b,
            self.classifier_c,
        ) = self.define_models(args_task_a, args_task_b, args_task_c)
        self.freeze_module(self.feature_extractor)
        self.optimizer = optimizer

        self.criterion_a = nn.CrossEntropyLoss()
        self.criterion_b = nn.CrossEntropyLoss()
        self.criterion_c = nn.CrossEntropyLoss()

        self.train_f1_a = MulticlassF1Score(num_classes=2, average="macro")
        self.train_f1_b = MulticlassF1Score(num_classes=4, average="macro")
        self.train_f1_c = MulticlassF1Score(num_classes=11, average="macro")

        self.val_f1_a = MulticlassF1Score(num_classes=2, average="macro")
        self.val_f1_b = MulticlassF1Score(num_classes=4, average="macro")
        self.val_f1_c = MulticlassF1Score(num_classes=11, average="macro")

        self.test_f1_a = MulticlassF1Score(num_classes=2, average="macro")
        self.test_f1_b = MulticlassF1Score(num_classes=4, average="macro")
        self.test_f1_c = MulticlassF1Score(num_classes=11, average="macro")

        self.train_loss_a = MeanMetric()
        self.val_loss_a = MeanMetric()
        self.test_loss_a = MeanMetric()

        self.train_loss_b = MeanMetric()
        self.val_loss_b = MeanMetric()
        self.test_loss_b = MeanMetric()

        self.train_loss_c = MeanMetric()
        self.val_loss_c = MeanMetric()
        self.test_loss_c = MeanMetric()

        self.val_f1_best_a = MaxMetric()
        self.val_f1_best_b = MaxMetric()
        self.val_f1_best_c = MaxMetric()

    def forward(self, input_ids, attention_mask):
        features = self.feature_extractor(input_ids, attention_mask).last_hidden_state
        logits_a = self.classifier_a(features)
        logits_b = self.classifier_b(features)
        logits_c = self.classifier_c(features)
        return logits_a, logits_b, logits_c

    def compute_losses(self, logits, labels):
        logits_a, logits_b, logits_c = logits
        labels_a, labels_b, labels_c = labels[:, 0], labels[:, 1], labels[:, 2]
        loss_a = self.criterion_a(logits_a, labels_a)
        loss_b = self.criterion_b(logits_b, labels_b)
        loss_c = self.criterion_c(logits_c, labels_c)
        return loss_a, loss_b, loss_c

    def model_step(self, batch):
        # Unpack the batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        losses = self.compute_losses(logits, labels)
        preds = self.apply_constraints(logits)

        return losses, preds, labels

    def training_step(self, batch, batch_idx):
        losses, preds, labels = self.model_step(batch)
        loss_a, loss_b, loss_c = losses
        preds_a, preds_b, preds_c = preds
        labels_a, labels_b, labels_c = labels[:, 0], labels[:, 1], labels[:, 2]
        opt_a, opt_b, opt_c = self.optimizers()

        # Log Task A
        self.train_loss_a(loss_a)
        self.train_f1_a(preds_a, labels_a)
        self.log("train/loss_a", self.train_loss_a, on_epoch=True, prog_bar=True)
        self.log("train/f1_a", self.train_f1_a, on_epoch=True, prog_bar=True)

        # Log Task B
        self.train_loss_b(loss_b)
        self.train_f1_b(preds_b, labels_b)
        self.log("train/loss_b", self.train_loss_b, on_epoch=True, prog_bar=True)
        self.log("train/f1_b", self.train_f1_b, on_epoch=True, prog_bar=True)

        # Log Task C
        self.train_loss_c(loss_c)
        self.train_f1_c(preds_c, labels_c)
        self.log("train/loss_c", self.train_loss_c, on_epoch=True, prog_bar=True)
        self.log("train/f1_c", self.train_f1_c, on_epoch=True, prog_bar=True)

        # Optimize
        opt_a.zero_grad()
        opt_b.zero_grad()
        opt_c.zero_grad()
        self.manual_backward(loss_a)
        self.manual_backward(loss_b)
        self.manual_backward(loss_c)
        opt_a.step()
        opt_b.step()
        opt_c.step()

        return {"loss_a": loss_a, "loss_b": loss_b, "loss_c": loss_c}

    def validation_step(self, batch, batch_idx):
        losses, preds, labels = self.model_step(batch)
        loss_a, loss_b, loss_c = losses
        preds_a, preds_b, preds_c = preds
        labels_a, labels_b, labels_c = labels[:, 0], labels[:, 1], labels[:, 2]

        # Log Task A
        self.val_loss_a(loss_a)
        self.val_f1_a(preds_a, labels_a)
        self.log("val/loss_a", self.val_loss_a, on_epoch=True, prog_bar=True)
        self.log("val/f1_a", self.val_f1_a, on_epoch=True, prog_bar=True)

        # Log Task B
        self.val_loss_b(loss_b)
        self.val_f1_b(preds_b, labels_b)
        self.log("val/loss_b", self.val_loss_b, on_epoch=True, prog_bar=True)
        self.log("val/f1_b", self.train_f1_b, on_epoch=True, prog_bar=True)

        # Log Task C
        self.val_loss_c(loss_c)
        self.val_f1_c(preds_c, labels_c)
        self.log("val/loss_c", self.val_loss_c, on_epoch=True, prog_bar=True)
        self.log("val/f1_c", self.val_f1_c, on_epoch=True, prog_bar=True)

        total_loss = loss_a + loss_b + loss_c
        self.log("val/loss", total_loss, on_epoch=True, prog_bar=True)
        return {"loss_a": loss_a, "loss_b": loss_b, "loss_c": loss_c}

    def test_step(self, batch, batch_idx):
        losses, preds, labels = self.model_step(batch)
        loss_a, loss_b, loss_c = losses
        preds_a, preds_b, preds_c = preds
        labels_a, labels_b, labels_c = labels[:, 0], labels[:, 1], labels[:, 2]

        # Log Task A
        self.test_loss_a(loss_a)
        self.test_f1_a(preds_a, labels_a)
        self.log("test/loss_a", self.test_loss_a, on_epoch=True, prog_bar=True)
        self.log("test/f1_a", self.test_f1_a, on_epoch=True, prog_bar=True)

        # Log Task B
        self.test_loss_b(loss_b)
        self.test_f1_b(preds_b, labels_b)
        self.log("test/loss_b", self.test_loss_b, on_epoch=True, prog_bar=True)
        self.log("test/f1_b", self.test_f1_b, on_epoch=True, prog_bar=True)

        # Log Task C
        self.test_loss_c(loss_c)
        self.test_f1_c(preds_c, labels_c)
        self.log("test/loss_c", self.test_loss_c, on_epoch=True, prog_bar=True)
        self.log("test/f1_c", self.test_f1_c, on_epoch=True, prog_bar=True)

        return {"loss_a": loss_a, "loss_b": loss_b, "loss_c": loss_c}

    def validation_epoch_end(self, outputs):
        f1_a = self.val_f1_a.compute()
        f1_b = self.val_f1_b.compute()
        f1_c = self.val_f1_c.compute()
        self.val_f1_best_a.update(f1_a)  # update best so far val f1 a
        self.val_f1_best_b.update(f1_b)  # update best so far val f1 b
        self.val_f1_best_c.update(f1_c)  # update best so far val f1 c
        self.log("val/f1_best_a", self.val_f1_best_a.compute(), on_epoch=True, prog_bar=True)
        self.log("val/f1_best_b", self.val_f1_best_b.compute(), on_epoch=True, prog_bar=True)
        self.log("val/f1_best_c", self.val_f1_best_c.compute(), on_epoch=True, prog_bar=True)

    @staticmethod
    def apply_constraints(logits):
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

        return preds_a, preds_b, preds_c

    def configure_optimizers(self):
        optimizer_a = self.optimizer(self.classifier_a.parameters(), lr=self.args.lr)
        optimizer_b = self.optimizer(self.classifier_b.parameters(), lr=self.args.lr)
        optimizer_c = self.optimizer(self.classifier_c.parameters(), lr=self.args.lr)
        return optimizer_a, optimizer_b, optimizer_c

    @staticmethod
    def define_models(args_task_a, args_task_b, args_task_c):
        feature_extractor = AutoModelForSequenceClassification.from_pretrained(
            args_task_a.model
        ).base_model

        classifier_a = TransformerModule.load_from_checkpoint(args_task_a.ckpt_path, args_task_a)
        classifier_b = TransformerModule.load_from_checkpoint(args_task_b.ckpt_path, args_task_b)
        classifier_c = TransformerModule.load_from_checkpoint(args_task_c.ckpt_path, args_task_c)
        return feature_extractor, classifier_a, classifier_b, classifier_c

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_module(module):
        for param in module.parameters():
            param.requires_grad = True
