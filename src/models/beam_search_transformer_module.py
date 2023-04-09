import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassF1Score
from transformers import AutoModelForSequenceClassification, get_scheduler


class BeamSearchTransformerModule(pl.LightningModule):
    def __init__(
        self,
        args,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.args = args
        (
            self.feature_extractor,
            self.classifier_a,
            self.classifier_b,
            self.classifier_c,
        ) = self.define_models(args)
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

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_f1_best_a = MaxMetric()
        self.val_f1_best_b = MaxMetric()
        self.val_f1_best_c = MaxMetric()

    @staticmethod
    def define_models(args):
        feature_extractor = AutoModelForSequenceClassification.from_pretrained(
            args.model
        ).base_model
        classifier_a = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2
        ).classifier
        classifier_b = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=4
        ).classifier
        classifier_c = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=11
        ).classifier
        return feature_extractor, classifier_a, classifier_b, classifier_c

    def forward(self, input_ids, attention_mask, labels=None, task=None):
        input_ids = input_ids.long()
        labels = labels.long()
        features = self.feature_extractor(input_ids, attention_mask=attention_mask)
        features = features.last_hidden_state
        logits_a = self.classifier_a(features)
        logits_b = self.classifier_b(features)
        logits_c = self.classifier_c(features)

        if labels is not None:
            loss_a, loss_b, loss_c = 0.0, 0.0, 0.0
            if task == "A":
                loss_a = self.criterion_a(logits_a, labels)
            elif task == "B":
                loss_b = self.criterion_b(logits_b, labels)
            elif task == "C":
                loss_c = self.criterion_c(logits_c, labels)
            return loss_a, loss_b, loss_c, logits_a, logits_b, logits_c
        else:
            return logits_a, logits_b, logits_c

    def _model_step(self, task_batch, task):
        # Unpack the batch
        input_ids = task_batch["input_ids"]
        attention_mask = task_batch["attention_mask"]
        labels = task_batch["labels"]

        loss_a, loss_b, loss_c, logits_a, logits_b, logits_c = self(
            input_ids, attention_mask, labels, task
        )

        if task == "A":
            loss = loss_a
            preds = self.beam_search(logits_a, logits_b, logits_c)[0]
            labels = labels
        elif task == "B":
            loss = loss_b
            preds = self.beam_search(logits_a, logits_b, logits_c)[1]
            labels = labels
        elif task == "C":
            loss = loss_c
            preds = self.beam_search(logits_a, logits_b, logits_c)[2]
            labels = labels

        return loss, preds, labels

    def beam_search(self, logits_a, logits_b, logits_c, beam_size=3):  # noqa: max-complexity: 13
        topk_a = torch.topk(logits_a, 2, dim=1)
        topk_b = torch.topk(logits_b, 4, dim=1)
        topk_c = torch.topk(logits_c, 11, dim=1)
        beam_result = []

        # Perform beam search
        for a_probs, b_probs, c_probs, a_indices, b_indices, c_indices in zip(
            topk_a.values,
            topk_b.values,
            topk_c.values,
            topk_a.indices,
            topk_b.indices,
            topk_c.indices,
        ):
            beam_probs = []
            for a_prob, a_idx in zip(a_probs, a_indices):
                if a_idx == 0:  # sexist
                    for b_prob, b_idx in zip(b_probs, b_indices):
                        if b_idx == 0:
                            for c_prob, c_idx in zip(c_probs[:2], c_indices[:2]):
                                beam_probs.append(
                                    (a_prob * b_prob * c_prob, (a_idx, b_idx, c_idx))
                                )
                        elif b_idx == 1:
                            for c_prob, c_idx in zip(c_probs[2:5], c_indices[2:5]):
                                beam_probs.append(
                                    (a_prob * b_prob * c_prob, (a_idx, b_idx, c_idx))
                                )
                        elif b_idx == 2:
                            for c_prob, c_idx in zip(c_probs[5:9], c_indices[5:9]):
                                beam_probs.append(
                                    (a_prob * b_prob * c_prob, (a_idx, b_idx, c_idx))
                                )
                        elif b_idx == 3:
                            for c_prob, c_idx in zip(c_probs[9:], c_indices[9:]):
                                beam_probs.append(
                                    (a_prob * b_prob * c_prob, (a_idx, b_idx, c_idx))
                                )
                else:  # not sexist
                    beam_probs.append((a_prob, (a_idx, 0, 0)))

            beam_probs.sort(reverse=True)
            beam_result.append(beam_probs[0][1])

        preds_a, preds_b, preds_c = zip(*beam_result)
        preds_a = torch.tensor(preds_a, device=logits_a.device)
        preds_b = torch.tensor(preds_b, device=logits_b.device)
        preds_c = torch.tensor(preds_c, device=logits_c.device)

        return preds_a, preds_b, preds_c

    def training_step(self, batch, batch_idx):
        task_a_batch, task_b_batch, task_c_batch = batch
        loss_a, loss_b, loss_c = 0.0, 0.0, 0.0
        # Task A
        loss_a, preds_a, labels_a = self._model_step(task_a_batch, "A")
        self.log("train/loss_a", loss_a)
        self.log("train/f1_a", self.train_f1_a(preds_a, labels_a))

        # Task B
        if len(task_b_batch["input_ids"]) > 0:
            loss_b, preds_b, labels_b = self._model_step(task_b_batch, "B")
            self.log("train/loss_b", loss_b)
            self.log("train/f1_b", self.train_f1_b(preds_b, labels_b))

        # Task C
        if len(task_c_batch["input_ids"]) > 0:
            loss_c, preds_c, labels_c = self._model_step(task_c_batch, "C")
            self.log("train/loss_c", loss_c)
            self.log("train/f1_c", self.train_f1_c(preds_c, labels_c))

        total_loss = loss_a + loss_b + loss_c
        self.log("train/loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        task_a_batch, task_b_batch, task_c_batch = batch
        loss_a, loss_b, loss_c = 0.0, 0.0, 0.0
        # Task A
        loss_a, preds_a, labels_a = self._model_step(task_a_batch, "A")
        self.log("val/loss_a", loss_a)
        self.log("val/f1_a", self.val_f1_a(preds_a, labels_a))

        # Task B
        if len(task_b_batch["input_ids"]) > 0:
            loss_b, preds_b, labels_b = self._model_step(task_b_batch, "B")
            self.log("val/loss_b", loss_b)
            self.log("val/f1_b", self.val_f1_b(preds_b, labels_b))

        # Task C
        if len(task_c_batch["input_ids"]) > 0:
            loss_c, preds_c, labels_c = self._model_step(task_c_batch, "C")
            self.log("val/loss_c", loss_c)
            self.log("val/f1_c", self.val_f1_c(preds_c, labels_c))

        total_loss = loss_a + loss_b + loss_c
        self.log("val/loss", total_loss)
        return {
            "loss": total_loss,
        }

    def test_step(self, batch, batch_idx):
        task_a_batch, task_b_batch, task_c_batch = batch
        loss_a, loss_b, loss_c = 0.0, 0.0, 0.0
        # Task A
        loss_a, preds_a, labels_a = self._model_step(task_a_batch, "A")
        self.log("test/loss_a", loss_a)
        self.log("test/f1_a", self.test_f1_a(preds_a, labels_a))

        # Task B
        if len(task_b_batch["input_ids"]) > 0:
            loss_b, preds_b, labels_b = self._model_step(task_b_batch, "B")
            self.log("test/loss_b", loss_b)
            self.log("test/f1_b", self.test_f1_b(preds_b, labels_b))

        # Task C
        if len(task_c_batch["input_ids"]) > 0:
            loss_c, preds_c, labels_c = self._model_step(task_c_batch, "C")
            self.log("test/loss_c", loss_c)
            self.log("test/f1_c", self.test_f1_c(preds_c, labels_c))

        total_loss = loss_a + loss_b + loss_c
        self.log("test/loss", total_loss)
        return total_loss

    def validation_epoch_end(self, outputs):
        f1_a = self.val_f1_a.compute()
        f1_b = self.val_f1_b.compute()
        f1_c = self.val_f1_c.compute()

        self.val_f1_best_a(f1_a)
        self.val_f1_best_b(f1_b)
        self.val_f1_best_c(f1_c)

        self.log("val/f1_best_a", self.val_f1_best_a.compute(), prog_bar=True)
        self.log("val/f1_best_b", self.val_f1_best_b.compute(), prog_bar=True)
        self.log("val/f1_best_c", self.val_f1_best_c.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.args.lr)
        num_training_steps = self.args.num_epoch * self.args.len_train_loader
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=self.args.n_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_module(module):
        for param in module.parameters():
            param.requires_grad = True
