import numpy as np
import pandas as pd
import hydra
from alive_progress import alive_bar

@hydra.main(config_path="config.yaml")
def process(cfg):
    pass

if __name__ == "__main__":
    process()