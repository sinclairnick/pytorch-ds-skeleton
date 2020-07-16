import hydra

@hydra.main(config_path="config.yaml")
def evaluate(cfg):
    pass

if __name__ == "__main__":
    evaluate()