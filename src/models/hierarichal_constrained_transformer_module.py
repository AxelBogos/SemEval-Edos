import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassF1Score
from transformers import AutoModelForSequenceClassification, get_scheduler


class HierarchicalTransformerModule(pl.LightningModule):
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

        self.criterion_a = nn.CrossEntropyLoss(
            weight=torch.tensor([0.6603, 2.0600], dtype=torch.float)
        )
        self.criterion_b = nn.CrossEntropyLoss(
            weight=torch.tensor([0.2641, 9.0323, 1.7610, 2.4034, 8.4084], dtype=torch.float)
        )
        self.criterion_c = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [
                    0.1100,
                    20.8333,
                    4.5932,
                    1.6272,
                    1.7335,
                    5.8333,
                    1.8315,
                    2.7978,
                    18.2292,
                    24.8227,
                    15.5556,
                    4.5220,
                ],
                dtype=torch.float,
            )
        )

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

    def forward(self, input_ids, attention_mask):
        # input_ids = input_ids.long()
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
        return {"loss": total_loss, "predictions": preds, "labels": labels}

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
        self.log("val/loss", total_loss)
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
