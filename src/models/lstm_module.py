from typing import Any, List

import pytorch_lightning as pl
import torch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassF1Score


class LSTMModule(pl.LightningModule):
    def __init__(
        self,
        args,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
    ):
        super().__init__()
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Embedding
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=args.len_vocab,
            embedding_dim=args.embedding_dim,
            padding_idx=args.pad_idx,
        )

        # Main network
        self.lstm = torch.nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = torch.nn.Dropout(args.dropout)
        fc_input_dim = 2 * args.hidden_dim if args.bidirectional else args.hidden_dim
        self.fc = torch.nn.Linear(fc_input_dim, args.num_target_class)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

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

    def forward(self, x):
        embeddings = self.embedding_layer(x)
        output, _ = self.lstm(embeddings)
        return self.fc(output[:, -1])

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_f1_best doesn't store f1 from these checks
        self.val_f1_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_f1(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_f1(preds, targets)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        f1 = self.val_f1.compute()  # get current val f1
        self.val_f1_best(f1)  # update best so far val f1
        # log `val_f1_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_f1(preds, targets)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.1)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
