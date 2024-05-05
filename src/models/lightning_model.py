import os
import torch
from torch.nn import functional as F
from torch import nn, Tensor
import torchmetrics
import lightning.pytorch as pl
import numpy as np

from settings import NUM_CLASSES, ALL_CLASSES


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        lr: float,
        l2_penalty: float,
        betas: tuple[float, float],
        scheduler_factor: float | None,
        scheduler_patience: int | None,
        upload_best_model: bool = True,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.lr = lr
        self.l2_penalty = l2_penalty
        self.betas = betas
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.upload_best_model = upload_best_model
        self.log_test = True

        # Metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.train_prec = torchmetrics.Precision(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.valid_prec = torchmetrics.Precision(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.test_prec = torchmetrics.Precision(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.test_recall = torchmetrics.Recall(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.valid_recall = torchmetrics.Recall(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.train_f1score = torchmetrics.F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.test_f1score = torchmetrics.F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.valid_f1score = torchmetrics.F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )

        self.valid_losses = []
        self.test_losses = []
        self.test_probabilities = []
        self.test_true_values = []
        self.test_class_ids = []

        # Model
        self.best_model_name = ""
        self.lowest_valid_loss = float("inf")
        self.lowest_valid_epoch: int | None = None
        self.using_best = False


    def _save_local(self):
        path = os.path.join(self.run_dir, f"epoch_{self.current_epoch}.pth")
        torch.save(self.state_dict(), path)

        return path

    def load_local(self, model_path: str):
        self.load_state_dict(torch.load(model_path))

    def load_best_model(self):
        self.load_local(self.best_model_name)
        self.using_best = True

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        return logits, loss

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, torch.optim.lr_scheduler.LRScheduler]]]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.l2_penalty,
            betas=self.betas,
        )
        if self.scheduler_patience is not None and self.scheduler_factor is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )
            return [optimizer], [{"scheduler": scheduler, "monitor": "validation/loss", "interval": "epoch", "frequency": 1}]
        return [optimizer], []

    def on_validation_epoch_start(self):
        self.valid_losses = []

    def on_test_epoch_start(self):
        self.test_losses = []
        self.test_probabilities = []
        self.test_true_values = []

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        xs, ys = batch
        preds, loss = self.loss(xs, ys)
        preds = torch.argmax(preds, 1)
        self.using_best = False

        self.log('train/loss', loss, on_epoch=True, on_step=True)
        self.train_acc(preds, ys)
        self.log('train/accuracy', self.train_acc, on_epoch=True, on_step=True)
        self.train_prec(preds, ys)
        self.log("train/precision", self.train_prec, on_epoch=True, on_step=True)
        self.train_recall(preds, ys)
        self.log("train/recall", self.train_recall, on_epoch=True, on_step=True)
        self.train_f1score(preds, ys)
        self.log("train/f1_score", self.train_f1score, on_epoch=True, on_step=True)
        self.log("train/epoch", self.current_epoch, on_epoch=False, on_step=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        suffix = ""
        if self.using_best:
            suffix = "_best"
        self.log(f"validation{suffix}/loss", loss, on_epoch=True, on_step=False)
        self.valid_acc(preds, ys)
        self.log(f'validation{suffix}/accuracy', self.valid_acc, on_epoch=True, on_step=False)
        self.valid_prec(preds, ys)
        self.log(f"validation{suffix}/precision", self.valid_prec, on_epoch=True,
                 on_step=False)
        self.valid_recall(preds, ys)
        self.log(f"validation{suffix}/recall", self.valid_recall, on_epoch=True,
                 on_step=False)
        self.valid_f1score(preds, ys)
        self.log(f"validation{suffix}/f1_score", self.valid_f1score, on_epoch=True,
                 on_step=False)
        self.log(f"validation{suffix}/epoch", self.current_epoch, on_epoch=False, on_step=True)

        self.valid_losses.append(loss.cpu())

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        if self.log_test:
            self.log(f"test/loss", loss, on_epoch=True, on_step=False)
            self.test_acc(preds, ys)
            self.log(f'test/accuracy', self.test_acc, on_epoch=True, on_step=False)
            self.test_prec(preds, ys)
            self.log("test/precision", self.test_prec, on_epoch=True, on_step=False)
            self.test_recall(preds, ys)
            self.log("test/recall", self.test_recall, on_epoch=True, on_step=False)
            self.test_f1score(preds, ys)
            self.log("test/f1_score", self.test_f1score, on_epoch=True, on_step=False)

        self.test_losses.append(loss.cpu())
        self.test_probabilities.append(torch.exp(logits))
        self.test_true_values.append(ys)

    def on_validation_epoch_end(self):
        if self.using_best:
            return
        path = self._save_local()

        avg_loss = np.mean(self.valid_losses)
        if avg_loss < self.lowest_valid_loss:
            self.lowest_valid_epoch = self.current_epoch
            self.lowest_valid_loss = avg_loss
            self.best_model_name = path

    def on_test_end(self):
        avg_loss = np.mean(self.test_losses)

        if self.upload_best_model:
            self._save_remote(
                self.model_name, epoch=self.lowest_valid_epoch, loss=avg_loss
            )

        flattened_probabilities = torch.flatten(
            torch.cat(self.test_probabilities)).view(-1, NUM_CLASSES).to(
            "cpu")
        self.test_class_ids = flattened_probabilities.argmax(dim=1)
        flattened_true_values = torch.flatten(torch.cat(self.test_true_values)).to(
            "cpu")

        if self.log_test:
            self.logger.experiment.log(
                {"test/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=flattened_probabilities,
                    y_true=flattened_true_values.numpy().tolist(),
                    class_names=ALL_CLASSES)}
            )
