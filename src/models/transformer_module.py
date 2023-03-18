import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassF1Score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


class TransformerModule(pl.LightningModule):
    def __init__(
        self,
        args,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(args.model, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, args.num_target_class)
        self.n_training_steps = args.n_training_steps
        self.n_warmup_steps = args.n_warmup_steps
        self.optimizer = optimizer
        self.scheduler.scheduler
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
        output = self.transformer(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        return output

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_f1_best doesn't store f1 from these checks
        self.val_f1_best.reset()

    def model_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        return loss, preds, outputs

    def training_step(self, batch, batch_idx):
        loss, preds, outputs = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_f1(preds, outputs)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "predictions": preds, "labels": outputs}

    def validation_step(self, batch, batch_idx):
        loss, preds, outputs = self.model_step(batch)

        self.val_loss(loss)
        self.val_f1(preds, outputs)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, outputs = self.model_step(batch)
        self.test_loss(loss)
        self.test_f1(preds, outputs)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))
