# This lab code assignment - PyTorch Pipelines & Model training - CIFAR -10

## CIFAR 10 IMAGE CLASSFICATION

### Project Overview: 


This project implements a complete Deep Learning pipeline in PyTorch with:

- Modular code structure

- Reproducible training

- DVC dataset tracking

- Multiple experiments with different hyperparameters

- Automatic metrics logging

The dataset used is CIFAR-10.

The focus is that the entire pipeline runs from:

" uv run python main.py "



### Project Structure:

```

ML-FRAMEWORKS-MLOPS25_LAB2/
│
├── .dvc/                          # DVC internal metadata and cache
│   ├── cache/                     # Stored dataset versions
│   ├── tmp/                       # Temporary DVC files
│   ├── config                     # DVC configuration
│   └── .gitignore                 # DVC ignore rules
│
├── .venv/                         # Local virtual environment (not committed to Git)
│
├── data/
│   ├── processed/                 # (Optional) processed dataset output
│   ├── raw/                       # Raw CIFAR-10 dataset (tracked with DVC)
│   │   ├── cifar-10-batches-py/   # Extracted dataset files
│   │   └── cifar-10-python.tar.gz # Original downloaded archive
│   └── raw.dvc                    # DVC tracking file (NOT raw data)
│
├── runs/                          # Experiment outputs
│   ├── exp1_baseline/
│   │   └── metrics.json           # Training & evaluation metrics
│   ├── exp2_low_LR/
│   │   └── metrics.json           # Metrics for low learning rate experiment
│   └── exp3_Big_Batch/
│       └── metrics.json           # Metrics for large batch size experiment
│
├── src/                           # Modular source code
│   ├── __pycache__/               # Python cache files
│   ├── config.py                  # Loads configuration from params.yaml
│   ├── dataset.py                 # Dataset download, transforms & DataLoader
│   ├── model.py                   # CNN model definition (nn.Module)
│   ├── train.py                   # Training and evaluation loops
│   └── utils.py                   # Seed setting and device selection
│
├── main.py                        # Entry point, runs full pipeline
├── params.yaml                    # Hyperparameters and experiment settings
│
├── pyproject.toml                 # Project dependencies and build configuration
├── uv.lock                        # Locked dependency versions for reproducibility
│
├── .gitignore                     # Git ignore rules
├── .dvcignore                     # DVC ignore rules
├── .python-version                # Python version specification
└── README.md                      # Project documentation

```


### Pipeline Architecture:


# Configuration

All hyperparameters are controlled through params.yaml and loaded using:

load_config() in config.py 

This ensures experiments are reproducible and centrally configured.



### Dataset Handling:


- CIFAR-10 is downloaded automatically if missing
- Custom transforms for training and evaluation
- DataLoader with configurable batch size and workers

Implemented in dataset.py



### Model:


A simple convolutional neural network:

- 3 Conv blocks
- ReLU activations
- MaxPooling
- Fully connected classifier
- Dropout for regularization

Implemented in model.py



### Training Loop:


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



### Reproducibility:


Reproducibility features:
- Fixed random seed
- Config-driven hyperparameters
- Device auto-detection (CPU / CUDA / MPS)
- Metrics saved per run

Each experiment saves results to:

" runs/<run_name>/metrics.json "



### DVC Dataset Tracking:


The dataset directory is tracked with DVC.

Raw data is not committed to Git.

To add dataset:

" dvc add data/raw "
" git add data/raw.dvc " 



### Experiments:


Three experiments were conducted using different hyperparameters.

| Experiment     | Epochs | Batch Size | Learning Rate | Final Test Accuracy     |
| -------------- | ------ | ---------- | ------------- | ----------------------- |
| exp1_baseline  | 5      | 128        | 0.001         | 0.796875                |
| exp2_low_LR    | 10     | 128        | 0.0005        | 0.796875                |
| exp3_Big_Batch | 10     | 256        | 0.001         | 0.77734375              |


Baseline final accuracy is taken from the last epoch in metrics.json.

You can modify this table with the exact results from your other runs.



### How to Run:


- Install dependencies
- Add dataset with DVC
- Run training

" uv run python main.py "

The experiment name is controlled in params.yaml:

run:
  name: exp1_baseline

  Changing this creates a new experiment folder inside runs/.




### Experiment Comparison & Analysis


# Experimental Setup:

All experiments were trained on CIFAR-10 using the same:

- Model architecture (SimpleCNN)
- Data augmentation strategy
- Optimizer (Adam)
- Number of epochs (5)

Only selected hyperparameters were modified to study their effect on performance.



### Results Summary:


| Experiment     | Batch Size | Learning Rate | Final Train Acc | Final Test Acc |
| -------------- | ---------- | ------------- | --------------- | -------------- |
| exp1_baseline  | 128        | 0.001         | 0.68936         | 0.796875       |
| exp2_low_LR    | 128        | 0.0005        | 0.73068         | 0.796875       |
| exp3_Big_Batch | 256        | 0.001         | 0.74132         | 0.77734375     |


Baseline results are taken from the last epoch of metrics.json.



### Analysis:


# 1. Effect of Learning Rate

Reducing the learning rate (exp2_low_LR) typically:
- Slows down convergence
- Produces more stable updates
-May require more epochs to reach the same performance

If test accuracy is lower than baseline after 5 epochs, this likely indicates under-training due to slower optimization. With more epochs, it might match or surpass baseline.

Key insight:
Learning rate strongly affects training speed more than final capacity of the model.



# 2. Effect of Batch Size

Increasing batch size (exp3_Big_Batch):

- Produces smoother gradient estimates
- Reduces gradient noise
- May generalize slightly worse if too large

If test accuracy drops slightly compared to baseline, this can be explained by reduced gradient noise, which sometimes helps generalization in smaller batches.

Key insight:
Larger batches often train faster per epoch but may reduce generalization slightly.



# 3. Overfitting Observation

Compare:

Train accuracy vs Test accuracy

If train accuracy is significantly higher than test accuracy:
→ Model may be overfitting.

In the baseline:
Train acc ≈ 0.70
Test acc ≈ 0.74

Since test accuracy is close to train accuracy, the model does not show severe overfitting within 5 epochs.



# 4. Convergence Behavior

From the baseline metrics:
- Loss consistently decreases
- Accuracy steadily increases

This indicates:
- Optimization is stable
- Learning rate is appropriate
- Model capacity is sufficient for early learning



### Overall Conclusion:

From these experiments we can conclude:
- Learning rate is the most sensitive hyperparameter for convergence speed.
- Batch size affects stability and possibly generalization.
- The SimpleCNN architecture can achieve around ~74% accuracy after 5 epochs.
- Further improvements would likely require:
More epochs
- Learning rate scheduling
- Data augmentation tuning
- Deeper architecture




### My way of looking at whats important from this lab conclussion is:



Learning Rate(LR) = How big steps you take.

Optimizer = Its the optimizer who takes the steps. It finds the lowest point.

Test_loss = Means its error = Not good if its high value because more errors.



Inotherwards the less the LR is the more precise the points will be to the weights which will give lowest loss error function.

One should always try to find optimal LR for ones data.

Also another point is test_acc isnt always good to look at because it can give misleading info about the performance for model prediction. Therefore concentrate on test_loss.
