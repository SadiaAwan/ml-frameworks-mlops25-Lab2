This lab code assignment - PyTorch Pipelines & Model training - CIFAR -10



Project Overview: 


This project implements a complete Deep Learning pipeline in PyTorch with:

- Modular code structure

- Reproducible training

- DVC dataset tracking

- Multiple experiments with different hyperparameters

- Automatic metrics logging

The dataset used is CIFAR-10.

The focus is that the entire pipeline runs from:

" uv run python main.py "



Project Structure:


.
├── data/
│   ├── raw/              # DVC tracked dataset
│   └── processed/
├── runs/                 # Experiment outputs
│   ├── exp1_baseline/
│   ├── exp2_low_LR/
│   └── exp3_Big_Batch/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── params.yaml
├── main.py
└── README.md



Pipeline Architecture:


Configuration

All hyperparameters are controlled through params.yaml and loaded using:

load_config() in config.py 

This ensures experiments are reproducible and centrally configured.



Dataset Handling:


- CIFAR-10 is downloaded automatically if missing
- Custom transforms for training and evaluation
- DataLoader with configurable batch size and workers

Implemented in dataset.py



Model:


A simple convolutional neural network:

- 3 Conv blocks
- ReLU activations
- MaxPooling
- Fully connected classifier
- Dropout for regularization

Implemented in model.py



Training Loop:


Training includes:
-  train_one_epoch
-  eval_loop
-  fit() wrapper for full training

Metrics logged per epoch:
- train_loss
- train_acc
- test_loss
- test_acc

Implemented in train.py 

Utility functions for reproducibility and device selection are in utils.py 



Reproducibility:


Reproducibility features:
- Fixed random seed
- Config-driven hyperparameters
- Device auto-detection (CPU / CUDA / MPS)
- Metrics saved per run

Each experiment saves results to:

" runs/<run_name>/metrics.json "



DVC Dataset Tracking:


The dataset directory is tracked with DVC.

Raw data is not committed to Git.

To add dataset:

" dvc add data/raw "
" git add data/raw.dvc " 



Experiments:


Three experiments were conducted using different hyperparameters.

| Experiment     | Epochs | Batch Size | Learning Rate | Final Test Accuracy     |
| -------------- | ------ | ---------- | ------------- | ----------------------- |
| exp1_baseline  | 5      | 128        | 0.001         | 0.742                   |
| exp2_low_LR    | 10     | 128        | 0.0001        | (fill with your result) |
| exp3_Big_Batch | 5      | 256        | 0.001         | (fill with your result) |


Baseline final accuracy is taken from the last epoch in metrics.json.

You can modify this table with the exact results from your other runs.



How to Run:


- Install dependencies
- Add dataset with DVC
- Run training

" uv run python main.py "

The experiment name is controlled in params.yaml:

run:
  name: exp1_baseline

  Changing this creates a new experiment folder inside runs/.

