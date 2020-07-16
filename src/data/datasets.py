from torch import Dataset

class TrainDataset(Dataset):
    
    def __init__(self, cfg):
        # Perform initializations
        pass
    
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        return (0, 0)
    
    def __len__(self):
        # Returns size of dataset
        return 0

class ValDataset(Dataset):
    
    def __init__(self, cfg):
        # Perform initializations
        path = cfg.training.data
        pass
    
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        return (0, 0)
    
    def __len__(self):
        # Returns size of dataset
        return 0

class TestDataset(Dataset):
    
    def __init__(self, cfg):
        # Perform initializations
        pass
    
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        return (0, 0)
    
    def __len__(self):
        # Returns size of dataset
        return 0