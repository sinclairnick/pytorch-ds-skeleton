import torchvision.transforms.transforms

class Augmentor():
    def __init__(self, cfg):
        self.augs = cfg.training.augmentations
        pass
    
    def augmentation1(self):
        pass