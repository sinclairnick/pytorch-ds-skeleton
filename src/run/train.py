import hydra
import torch
from torch.utils.data import DataLoader

from model.lightning import LightningModel
from pytorch_lightning import Traner, seed_everything
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path="config.yaml")
def train(cfg):

    logger = WandbLogger(project=cfg.project_name, config=cfg) if cfg.training.log else None

    model = LightningModel(cfg)
    
    trainer = Trainer(
        logger=logger,
        gpus=cfg.training.gpus,
        precision=cfg.training.precision,
        auto_scale_batch_size=cfg.training.hparams.auto_scale_batch_size,
        deterministic=cfg.training.reproducability
        )

    if cfg.training.reproducability > 0:
        seed_everything(1000) 

    trainer.fit(model)
    

if __name__ == "__main__":
    train()