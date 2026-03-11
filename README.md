# DiversifyAnalysis

Replication of the experiments from the ICLR 2023 paper:
**"Out-of-distribution Representation Learning for Time Series Classification"** (Lu et al., 2023)

This project was completed as part of the **AI and Cybersecurity** course at the **University of Luxembourg**.

## What This Project Is About

Time series classification models often fail when test data comes from a different distribution than training data — a common problem in real-world deployments. This paper proposes a representation learning approach (Diversify) that improves model robustness against out-of-distribution (OOD) data.

This repo reproduces the core experiments from the paper using the EMG (Electromyography) dataset to validate the authors' findings on OOD generalisation for time series classification.

## What I Did

- Reproduced the preprocessing, training, and evaluation pipeline from the original paper
- Ran experiments on the EMG dataset using the Diversify algorithm
- Validated the OOD representation learning approach and analysed model accuracy across test groups

## Key Concepts

- **Out-of-Distribution (OOD) Learning** — training models to generalise beyond the training distribution
- **Time Series Classification** — classifying sequential data (e.g. sensor readings, network traffic, physiological signals)
- **Representation Learning** — learning meaningful feature representations that improve generalisation
- **Diversify Algorithm** — latent domain-based approach to improve OOD robustness

## How to Run

### 1. Prerequisites

```bash
pip install -r requirements.txt
```

### 2. Preprocess Dataset

Download and prepare the EMG dataset:

```bash
python prepare_dataset.py --data_dir ./data/ --dataset EMG
```

If executed successfully, the dataset will be saved to `./data/emg`

### 3. Train

```bash
python train.py --data_dir ./data/ --task cross_people --test_envs 0 --dataset emg --algorithm diversify --latent_domain_num 10 --alpha1 1.0 --alpha 1.0 --lam 0.0 --local_epoch 3 --max_epoch 5 --lr 0.01 --output ./data/train_output/act/cross_people-emg-Diversify-0-10-1-1-0-3-50-0.01 --output_model ./data/model_output
```

**Key parameters:**
- `--data_dir` — path to your dataset directory
- `--dataset` — dataset name (e.g. `emg`)
- `--test_envs` — specify a different test group if applicable
- `--output` — training log will be saved here
- `--output_model` — generated models will be saved here
- Run `--help` to see the full parameter list

### 4. Evaluate

For each training round, the output value `target acc` is the accuracy value for the model.

## Technologies Used

- Python
- PyTorch
- NumPy / Pandas
- Machine Learning / OOD Representation Learning
- Time Series Classification

## Course Context

This replication was completed as part of the **AI and Cybersecurity** course at the University of Luxembourg, exploring the intersection of machine learning robustness and security-relevant applications such as anomaly detection and out-of-distribution generalisation in sensor and network data.

## Reference

```
@inproceedings{lu2022out,
  title={Out-of-distribution Representation Learning for Time Series Classification},
  author={Lu, Wang and Wang, Jindong and Sun, Xinwei and Chen, Yiqiang and Xie, Xing},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
