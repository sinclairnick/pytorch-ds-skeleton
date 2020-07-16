import hydra
from model.lightning import LightningModule

@hydra.main(config_path="config.yaml")
def inference(cfg):
    model = LightningModule(cfg)
    x = [] # get data
    model(x)


if __name__ == "__main__":
    inference()