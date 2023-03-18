import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassF1Score
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_scheduler,
)


class TransformerModule(pl.LightningModule):
    def __init__(
        self,
        args,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.args = args
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=args.num_target_class
        )
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

        # metric objects for calculating and macro f1 across batches
        self.train_f1 = MulticlassF1Score(num_classes=args.num_target_class, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=args.num_target_class, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=args.num_target_class, average="macro")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation f1
        self.val_f1_best = MaxMetric()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(**input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def on_train_start(self):
        self.val_f1_best.reset()

    def model_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_f1(preds, labels)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "predictions": preds, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.model_step(batch)

        self.val_loss(loss)
        self.val_f1(preds, labels)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.model_step(batch)
        self.test_loss(loss)
        self.test_f1(preds, labels)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.args.lr)
        num_training_steps = self.args.num_epoch * self.args.len_train_loader
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))
