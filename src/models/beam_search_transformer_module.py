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
        self.criterion = nn.CrossEntropyLoss()

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

    def define_models(self, args):
        feature_extractor = AutoModelForSequenceClassification.from_pretrained(
            args.model
        ).base_model
        classifier_a = AutoModelForSequenceClassification.from_pretrained(args.model).classifier
        classifier_b = AutoModelForSequenceClassification.from_pretrained(args.model).classifier
        classifier_c = AutoModelForSequenceClassification.from_pretrained(args.model).classifier
        return feature_extractor, classifier_a, classifier_b, classifier_c

    def forward(self, input_ids, attention_mask, labels=None):
        features = self.feature_extractor(input_ids, attention_mask=attention_mask)
        logits_a = self.classifier_a(features)
        logits_b = self.classifier_b(features)
        logits_c = self.classifier_c(features)

        if labels is not None:
            loss_a = self.criterion(logits_a, labels[:, 0])
            loss_b = self.criterion(logits_b, labels[:, 1])
            loss_c = self.criterion(logits_c, labels[:, 2])
            return loss_a, loss_b, loss_c, logits_a, logits_b, logits_c
        else:
            return logits_a, logits_b, logits_c

    def _model_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss_a, loss_b, loss_c, logits_a, logits_b, logits_c = self(
            input_ids, attention_mask, labels
        )

        preds_a, preds_b, preds_c = self.beam_search(logits_a, logits_b, logits_c)

        total_loss = loss_a + loss_b + loss_c
        labels_a, labels_b, labels_c = labels[:, 0], labels[:, 1], labels[:, 2]

        return total_loss, preds_a, preds_b, preds_c, labels_a, labels_b, labels_c

    def beam_search(self, logits_a, logits_b, logits_c):  # noqa: max-complexity: 13
        topk_a = torch.topk(logits_a, 2, dim=1)
        topk_b = torch.topk(logits_b, 4, dim=1)
        topk_c = torch.topk(logits_c, 11, dim=1)

        beam_result = []
        for a_probs, b_probs, c_probs in zip(topk_a.values, topk_b.values, topk_c.values):
            beam_probs = []
            for a_prob, a_idx in zip(a_probs, topk_a.indices):
                if a_idx == 1:  # sexist
                    for b_prob, b_idx in zip(b_probs, topk_b.indices):
                        if b_idx == 1:
                            for c_prob, c_idx in zip(c_probs[:2], topk_c.indices[:2]):
                                beam_probs.append(
                                    (a_prob * b_prob * c_prob, (a_idx, b_idx, c_idx))
                                )
                        elif b_idx == 2:
                            for c_prob, c_idx in zip(c_probs[2:5], topk_c.indices[2:5]):
                                beam_probs.append(
                                    (a_prob * b_prob * c_prob, (a_idx, b_idx, c_idx))
                                )
                        elif b_idx == 3:
                            for c_prob, c_idx in zip(c_probs[5:9], topk_c.indices[5:9]):
                                beam_probs.append(
                                    (a_prob * b_prob * c_prob, (a_idx, b_idx, c_idx))
                                )
                        elif b_idx == 4:
                            for c_prob, c_idx in zip(c_probs[9:], topk_c.indices[9:]):
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

        # Task A
        loss_a, preds_a, _, _, labels_a, _, _ = self._model_step(task_a_batch)
        self.train_loss(loss_a)
        self.train_f1_a(preds_a, labels_a)

        # Task B
        _, loss_b, preds_b, _, _, labels_b, _ = self._model_step(task_b_batch)
        self.train_loss(loss_b)
        self.train_f1_b(preds_b, labels_b)

        # Task C
        _, _, loss_c, preds_c, _, _, labels_c = self._model_step(task_c_batch)
        self.train_loss(loss_c)
        self.train_f1_c(preds_c, labels_c)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1_a", self.train_f1_a, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1_b", self.train_f1_b, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1_c", self.train_f1_c, on_step=True, on_epoch=True, prog_bar=True)

        total_loss = loss_a + loss_b + loss_c
        return {
            "loss": total_loss,
            "predictions_a": preds_a,
            "labels_a": labels_a,
            "predictions_b": preds_b,
            "labels_b": labels_b,
            "predictions_c": preds_c,
            "labels_c": labels_c,
        }

    def validation_step(self, batch, batch_idx):
        task_a_batch, task_b_batch, task_c_batch = batch

        # Task A
        loss_a, preds_a, _, _, labels_a, _, _ = self._model_step(task_a_batch)
        self.val_loss(loss_a)
        self.val_f1_a(preds_a, labels_a)

        # Task B
        _, loss_b, preds_b, _, _, labels_b, _ = self._model_step(task_b_batch)
        self.val_loss(loss_b)
        self.val_f1_b(preds_b, labels_b)

        # Task C
        _, _, loss_c, preds_c, _, _, labels_c = self._model_step(task_c_batch)
        self.val_loss(loss_c)
        self.val_f1_c(preds_c, labels_c)

        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/f1_a", self.val_f1_a, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/f1_b", self.val_f1_b, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/f1_c", self.val_f1_c, on_step=True, on_epoch=True, prog_bar=True)

        total_loss = loss_a + loss_b + loss_c
        return {
            "loss": total_loss,
            "predictions_a": preds_a,
            "labels_a": labels_a,
            "predictions_b": preds_b,
            "labels_b": labels_b,
            "predictions_c": preds_c,
            "labels_c": labels_c,
        }

    def test_step(self, batch, batch_idx):
        task_a_batch, task_b_batch, task_c_batch = batch

        # Task A
        loss_a, preds_a, _, _, labels_a, _, _ = self._model_step(task_a_batch)
        self.test_loss(loss_a)
        self.test_f1_a(preds_a, labels_a)

        # Task B
        _, loss_b, preds_b, _, _, labels_b, _ = self._model_step(task_b_batch)
        self.test_loss(loss_b)
        self.test_f1_b(preds_b, labels_b)

        # Task C
        _, _, loss_c, preds_c, _, _, labels_c = self._model_step(task_c_batch)
        self.test_loss(loss_c)
        self.test_f1_c(preds_c, labels_c)

        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/f1_a", self.test_f1_a, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/f1_b", self.test_f1_b, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/f1_c", self.test_f1_c, on_step=True, on_epoch=True, prog_bar=True)

        total_loss = loss_a + loss_b + loss_c
        return {
            "loss": total_loss,
            "predictions_a": preds_a,
            "labels_a": labels_a,
            "predictions_b": preds_b,
            "labels_b": labels_b,
            "predictions_c": preds_c,
            "labels_c": labels_c,
        }

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