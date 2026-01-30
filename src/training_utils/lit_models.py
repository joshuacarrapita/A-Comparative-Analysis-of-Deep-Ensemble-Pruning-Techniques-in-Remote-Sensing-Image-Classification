# stdlib imports
from typing import Optional

# third-party imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.classification import MulticlassConfusionMatrix

# local imports
from configilm import ConfigILM

# ===========================================================================
# LightningModule Class Definition
# ===========================================================================
class LitImageEncoder(pl.LightningModule):
    """
    Wrapper around a pytorch module, allowing this module to be used in automatic
    training with Pytorch Lightning.
    Among other things, the wrapper allows us to do automatic training and removes the
    need to manage data on different devices (e.g. GPU and CPU).
    """
    def __init__(
        self,
        config: ConfigILM.ILMConfiguration,
        lr: float = 5e-4,
        loss_fn: callable = F.cross_entropy,
        class_weights: Optional[torch.Tensor] = None,
        optimizer: callable = torch.optim.AdamW,
    ):
        super().__init__()

        # hyperparameters 
        self.lr = lr
        self.config = config
        self.model = ConfigILM.ConfigILM(config)

        self.loss_fn = loss_fn
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
            
        self.optimizer = optimizer
        self.save_hyperparameters() 

        # metrics
        self.train_cm = MulticlassConfusionMatrix(num_classes=self.config.classes)
        self.val_cm = MulticlassConfusionMatrix(num_classes=self.config.classes)
        self.test_cm = MulticlassConfusionMatrix(num_classes=self.config.classes)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, weight_decay=0.01) 
        return optimizer
    
    def forward(self, batch):
        # because we are a wrapper, we call the inner function manually
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        # retrieve batch data
        img, label = batch
        label = torch.argmax(label, dim=1) # one-hot -> index

        # infer input and calculate loss
        logits = self.model(img)
        loss = self.loss_fn(logits,label,weight=self.class_weights)

        # calculate batch accuracy
        preds = torch.argmax(logits,dim=1) # logits -> index
        acc = 100 * (preds == label).float().mean()

        # update training confusion matrix
        self.train_cm.update(preds,label)
        
        # log metrics to logger
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        # return loss for optimizer to work with
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # retrieve batch data
        img, label = batch
        label = torch.argmax(label, dim=1) # one-hot -> index

        # infer input and calculate loss
        logits = self.model(img)
        loss = self.loss_fn(logits,label,weight=self.class_weights)

        # calculate batch accuracy
        preds = torch.argmax(logits,dim=1) # logits -> index
        acc = 100 * (preds == label).float().mean()

        # update validation confusion matrix
        self.val_cm.update(preds,label) 

        # log metrics to logger
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True) 
       
    def predict_step(self, batch, batch_idx):
        # retrieve batch data
        img, label = batch
        label = torch.argmax(label, dim=1) # one-hot -> index

        # infer input and calculate probas
        logits = self.model(img)
        probas = torch.nn.functional.softmax(logits, dim=1)

        return {'prediction_results': probas, 'target_labels': label}
        
    def test_step(self, batch, batch_idx):
        img, label = batch
        label = torch.argmax(label, dim=1) # one-hot -> index

        # infer input and calculate loss
        logits = self.model(img)
        loss = self.loss_fn(logits,label,weight=self.class_weights)

        # calculate batch accuracy
        preds = torch.argmax(logits,dim=1) # logits -> index
        acc = 100 * (preds == label).float().mean()

        # update test confusion matrix
        self.test_cm.update(preds,label)

        # log metrics to logger
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True) 

    def on_train_epoch_end(self):
        cm = self.train_cm.compute().detach().cpu()
        self._log_confusion_matrix(cm, tag='train_cm')

        # calc and log per-class accuracy
        per_class_acc = self._calc_per_class_accuracy(cm) # returns (C,)
        self._log_class_accuracy(per_class_acc, tag='train_per_class_acc')

        # calc and log average per-class accuracy
        avg_class_acc = torch.mean(per_class_acc)
        self.logger.experiment.add_scalar('train_avg_class_acc', avg_class_acc, self.current_epoch)

        self.train_cm.reset()

    def on_validation_epoch_end(self):
        # calc and log confusion matrix
        cm = self.val_cm.compute().detach().cpu()
        self._log_confusion_matrix(cm, tag='val_cm')

        # calc and log per-class accuracy
        per_class_acc = self._calc_per_class_accuracy(cm) # returns (C,)
        self._log_class_accuracy(per_class_acc, tag='val_per_class_acc')

        # calc and log average per-class accuracy
        avg_class_acc = torch.mean(per_class_acc)
        self.logger.experiment.add_scalar('val_avg_class_acc', avg_class_acc, self.current_epoch)

        self.val_cm.reset()

    def on_test_epoch_end(self):
        # calc and log confusion matrix
        cm = self.test_cm.compute().detach().cpu()
        self._log_confusion_matrix(cm, tag='test_cm')

        # calc and log per-class accuracy
        per_class_acc = self._calc_per_class_accuracy(cm) # returns (C,)
        self._log_class_accuracy(per_class_acc, tag='test_per_class_acc')

        # calc and log average per-class accuracy
        avg_class_acc = torch.mean(per_class_acc)
        self.log('test_avg_class_acc', avg_class_acc, on_step=False, on_epoch=True)
        self.logger.experiment.add_scalar('test_avg_class_acc', avg_class_acc, self.current_epoch)

        self.test_cm.reset()

    def _calc_per_class_accuracy(self, cm_tensor):
        """
            cm_tensor: (C, C) detached int64 confusion matrix
            Returns: (C,) float tensor with TP/(TP+FN) for each class
        """
        # true positives = diagonal, total true instances = row sum
        tp = torch.diag(cm_tensor).float()
        total = cm_tensor.sum(dim=1).float()
        # avoid division‑by‑zero
        return torch.where(total == 0, torch.zeros_like(tp), tp / total)
        
    def _log_confusion_matrix(self, cm_tensor, tag: str):
        """
            :param cm_tensor : (C, C) detached int64 tensor of counts 
            :param tag       : logger prefix (e.g., "val_cm")
        """
        # convert confusion matrix to numpy
        cm_np = cm_tensor.numpy()

        # log raw counts
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(
            cm_np,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title(f"Raw CM (Counts) - {tag.split('_')[0].upper()}")
        
        # save with _raw suffix
        self.logger.experiment.add_figure(f"{tag}_raw", fig, self.current_epoch)
        plt.close(fig)

        # log row-normalized (recall) 
        # normalize by true labels 
        row_sums = cm_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 # avoid div by zero
        cm_norm = cm_np / row_sums

        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f", # Float formatting
            cmap="Blues",
            vmin=0.0, vmax=1.0,
            cbar=True,
            ax=ax,
        )
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title(f"Normalized CM (Recall) - {tag.split('_')[0].upper()}")

        # save with _norm suffix
        self.logger.experiment.add_figure(f"{tag}_norm", fig, self.current_epoch)
        plt.close(fig)

    def _log_class_accuracy(self, acc_tensor, tag: str):
        """Log a 1×C detached accuracy tensor as a seaborn heat‑map."""
        acc_tensor = acc_tensor.numpy().reshape(1, -1)   # (1, C)

        labels = [
            (lbl[:20] + "...") if len(lbl) > 20 else lbl 
            for lbl in self.config.class_names
        ]

        fig, ax = plt.subplots(figsize=(max(6, self.config.classes * 0.4), 3))
        sns.heatmap(
            acc_tensor,
            annot=True,
            annot_kws={"rotation": 90, "ha": "center", "va": "center"},
            fmt=".2%",
            cmap="viridis",
            cbar=False,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xticks(np.arange(self.config.classes) + 0.5)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks([])  # hide the single y‑tick
        ax.set_title("Per‑class Accuracy (validation)")
        plt.tight_layout()
        self.logger.experiment.add_figure(tag, fig, self.current_epoch)
        plt.close(fig)
