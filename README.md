## Skeleton How To

This skeleton uses PyTorch-Lightning and Hydra with common DS libraries like numpy and pandas preinstalled. It is set up for Google Cloud Storage by default, but this can be easily changed.

Install dependencies using conda

```
conda create -f environment.yaml
```

File Structure:

```bash
├── README.md
├── bin
│   ├── checkpoints # store model checkpoints here
│   └── models # store model weights and config here
├── config
│   ├── config.yaml # the root config file which orchestrates the below subconfigs
│   ├── data # raw data sources
│   │   └── data.yaml
│   ├── inference
│   │   └── inference.yaml
│   ├── model # model settings
│   │   └── model_name.yaml
│   ├── testing # model testing
│   │   └── testing.yaml
│   └── training # training parameters
│       └── training.yaml
├── data
│   ├── processed # process the raw data and place here
│   └── raw # only to be used for immutable data
├── environment.yaml # run `conda create -f environment.yaml` to create the virtual environment
├── logs
├── notebooks
│   └── explore_data.ipynb
└── src
    ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   ├── augment.py # class encapsulating data augmentation methods
    │   ├── datasets.py # classes for retrieving train, val, test data
    │   ├── process.py # used to process raw data
    │   └── retrieve.py # retrieve the data from remote sources
    ├── model
    │   ├── __init__.py
    │   ├── lightning.py # wrapper encapsulating data retrieval, loading, augmentation and model training, val, testing
    │   └── model.py # defines the model structure
    ├── run
    │   ├── __init__.py
    │   ├── evaluate.py # runs the model on the test set
    │   ├── inference.py # runs the model on specified inferencing data
    │   └── train.py # runs model on training data
    └── viz
        ├── __init__.py
        └── visualize.py # used for producing graphs and visualizations
```

**<sub><sup>Delete this message before publishing your own repo</sup></sub>**

---

# Project Name

## Description

A Pytorch project used for ...

## How to run

#### Set up conda environment

```
conda env create -f environment.yaml
```

#### Run script

```
python ...
```

## Results

The results show ...
