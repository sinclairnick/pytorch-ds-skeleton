import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import gdown

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule

from model.model import Model
from data.datasets import TrainDataset, ValDataset, TestDataset

class LightningTemplateModel(LightningModule):
    """
    Sample model demonstrating order of method call by lightning (where applicable)
    """

    def __init__(self, cfg):
        # Init superclass
        super().__init__()
        self.cfg = cfg
        self.hparams = cfg.training.hparams
        self.model = Model(**cfg.model.hparams)

    def prepare_data(self):
        """
        Run only at the start of training.
        Download and save data to disk
        """
        # Iterate over and download all the Google Drive documents
        cfg = self.cfg
        if cfg.data.gdrive is not None and isinstance(cfg.data.gdrive, list):
            for file_id, fname in cfg.data.gdrive.items():
                url = "https://drive.google.com/uc?id={id}".format(id=file_id)
                gdown.cached_download(url, fname, postprocess=gdown.extractall)

    def setup(self):
        """
        Perform operations like getting number of classes, setting anchor boxes, etc.
        """
        pass

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # Perform preprocessing
        return DataLoader(
            TrainDataset(self.cfg),
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers
            )

    def val_dataloader(self):
        return DataLoader(
            ValDataset(self.cfg),
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers
            )

    def test_dataloader(self):
        return DataLoader(
            TestDataset(self.cfg),
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers
            )

    def forward(self, x):
        """
        Run the inputs through the model
        """
        return self.model(x)


    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss }

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this when `trainer.test()` is called
        """
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs. `log` is passed to the configured logger.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss, 'log': logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        logs = {'test_loss': avg_loss, 'test_acc': test_acc}
        return {'test_loss': avg_loss, 'log': logs}
