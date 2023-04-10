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
        self.train_f1_b = MulticlassF1Score(num_classes=5, average="macro")
        self.train_f1_c = MulticlassF1Score(num_classes=12, average="macro")

        self.val_f1_a = MulticlassF1Score(num_classes=2, average="macro")
        self.val_f1_b = MulticlassF1Score(num_classes=5, average="macro")
        self.val_f1_c = MulticlassF1Score(num_classes=12, average="macro")

        self.test_f1_a = MulticlassF1Score(num_classes=2, average="macro")
        self.test_f1_b = MulticlassF1Score(num_classes=5, average="macro")
        self.test_f1_c = MulticlassF1Score(num_classes=12, average="macro")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_f1_best_a = MaxMetric()
        self.val_f1_best_b = MaxMetric()
        self.val_f1_best_c = MaxMetric()

    def on_train_start(self):
        self.val_f1_best_a.reset()
        self.val_f1_best_b.reset()
        self.val_f1_best_c.reset()

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.long()
        features = self.feature_extractor(input_ids, attention_mask=attention_mask)
        features = features.last_hidden_state
        logits_a = self.classifier_a(features)
        logits_b = self.classifier_b(features)
        logits_c = self.classifier_c(features)

        if labels is not None:
            labels_a, labels_b, labels_c = labels[:, 0], labels[:, 1], labels[:, 2]
            loss_a = self.criterion_a(logits_a, labels_a)
            loss_b = self.criterion_b(logits_b, labels_b)
            loss_c = self.criterion_c(logits_c, labels_c)
            return loss_a, loss_b, loss_c, logits_a, logits_b, logits_c
        return logits_a, logits_b, logits_c

    def model_step(self, batch):
        # Unpack the batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss_a, loss_b, loss_c, logits_a, logits_b, logits_c = self(
            input_ids, attention_mask, labels
        )

        preds_a = self.beam_search(logits_a, logits_b, logits_c)[0]
        preds_b = self.beam_search(logits_a, logits_b, logits_c)[1]
        preds_c = self.beam_search(logits_a, logits_b, logits_c)[2]

        losses = (loss_a, loss_b, loss_c)
        preds = (preds_a, preds_b, preds_c)
        return losses, preds, labels

    def beam_search(self, logits_a, logits_b, logits_c, epsilon=0.3):
        batch_size = logits_a.size(0)
        probs_a = torch.softmax(logits_a, dim=1)
        probs_b = torch.softmax(logits_b, dim=1)
        probs_c = torch.softmax(logits_c, dim=1)

        preds_a = torch.argmax(probs_a, dim=1)

        # Apply conditioning on logits_b
        mask_b = torch.tensor(
            [[1.0, epsilon, epsilon, epsilon, epsilon], [epsilon, 1.0, 1.0, 1.0, 1.0]]
        ).to(logits_b.device)
        conditioned_probs_b = probs_b * mask_b[preds_a]

        joint_probs_ab = torch.unsqueeze(probs_a, 2) * torch.unsqueeze(conditioned_probs_b, 1)
        preds_b = torch.argmax(joint_probs_ab.view(batch_size, -1), dim=1) % logits_b.size(1)

        # Apply conditioning on logits_c
        mask_c = torch.tensor(
            [
                [
                    1.0,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                ],
                [
                    epsilon,
                    1.0,
                    1.0,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                ],
                [
                    epsilon,
                    epsilon,
                    epsilon,
                    1.0,
                    1.0,
                    1.0,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                ],
                [
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    epsilon,
                    epsilon,
                ],
                [
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    epsilon,
                    1.0,
                    1.0,
                ],
            ]
        ).to(logits_c.device)
        conditioned_probs_c = probs_c * mask_c[preds_b]

        joint_probs_abc = torch.unsqueeze(joint_probs_ab, 3) * torch.unsqueeze(
            conditioned_probs_c, 1
        ).unsqueeze(2)
        preds_c = torch.argmax(joint_probs_abc.view(batch_size, -1), dim=1) % logits_c.size(1)

        return preds_a, preds_b, preds_c

    def training_step(self, batch, batch_idx):
        losses, preds, labels = self.model_step(batch)
        loss_a, loss_b, loss_c = losses
        preds_a, preds_b, preds_c = preds
        labels_a, labels_b, labels_c = labels[:, 0], labels[:, 1], labels[:, 2]

        # Log Task A
        self.log("train/loss_a", loss_a)
        self.log("train/f1_a", self.train_f1_a(preds_a, labels_a))

        # Log Task B
        self.log("train/loss_b", loss_b)
        self.log("train/f1_b", self.train_f1_b(preds_b, labels_b))

        # Log Task C
        self.log("train/loss_c", loss_c)
        self.log("train/f1_c", self.train_f1_c(preds_c, labels_c))

        total_loss = loss_a + loss_b + loss_c
        self.log("train/total_loss", total_loss)
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        losses, preds, labels = self.model_step(batch)
        loss_a, loss_b, loss_c = losses
        preds_a, preds_b, preds_c = preds
        labels_a, labels_b, labels_c = labels[:, 0], labels[:, 1], labels[:, 2]

        # Log Task A
        self.log("val/loss_a", loss_a)
        self.log("val/f1_a", self.train_f1_a(preds_a, labels_a))

        # Log Task B
        self.log("val/loss_b", loss_b)
        self.log("val/f1_b", self.train_f1_b(preds_b, labels_b))

        # Log Task C
        self.log("val/loss_c", loss_c)
        self.log("val/f1_c", self.train_f1_c(preds_c, labels_c))

        total_loss = loss_a + loss_b + loss_c
        self.log("val/total_loss", total_loss)
        return {"loss": total_loss}

    def test_step(self, batch, batch_idx):
        losses, preds, labels = self.model_step(batch)
        loss_a, loss_b, loss_c = losses
        preds_a, preds_b, preds_c = preds
        labels_a, labels_b, labels_c = labels[:, 0], labels[:, 1], labels[:, 2]

        # Log Task A
        self.log("test/loss_a", loss_a)
        self.log("test/f1_a", self.train_f1_a(preds_a, labels_a))

        # Log Task B
        self.log("test/loss_b", loss_b)
        self.log("test/f1_b", self.train_f1_b(preds_b, labels_b))

        # Log Task C
        self.log("test/loss_c", loss_c)
        self.log("test/f1_c", self.train_f1_c(preds_c, labels_c))

        total_loss = loss_a + loss_b + loss_c
        self.log("test/total_loss", total_loss)
        return {"loss": total_loss}

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
    def define_models(args):
        feature_extractor = AutoModelForSequenceClassification.from_pretrained(
            args.model
        ).base_model
        classifier_a = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2
        ).classifier
        classifier_b = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=5
        ).classifier
        classifier_c = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=12
        ).classifier
        return feature_extractor, classifier_a, classifier_b, classifier_c

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_module(module):
        for param in module.parameters():
            param.requires_grad = True
