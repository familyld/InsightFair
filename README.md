# InsightFair
Author's implementation of InsightFair in "What Hides behind Unfairness? Exploring Dynamics Fairness in Reinforcement Learning". Our code is built off of [Omnisafe](https://github.com/PKU-Alignment/omnisafe/tree/main).

Link to our paper:
- IJCAI: TBD
- arXiv: [https://arxiv.org/abs/2404.10942](https://arxiv.org/abs/2404.10942)

## Prerequisites

- PyTorch 2.0.1 with Python 3.9 
- gymnasium 0.29.0
- numba 0.57.0

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

## Usage

For training `ALGO` on `ENV` (e.g. `Allocation-v0` or `Lending-v0`), run:

```
python main.py --env_name=Envname --version=VersionName --gpu=0 
```

The results are collected in `./scripts/runs/{ALGO}-{ENV}/`, where

- `progress.csv` records the log data in the `.csv` format for analysis purpose,
- `config.json` records the settings and hyperparameters.

# Bibtex

```
@inproceedings{deng2024hides,
  title={What Hides behind Unfairness? Exploring Dynamics Fairness in Reinforcement Learning},
  author={Deng, Zhihong and Jiang, Jing and Long, Guodong and Zhang, Chengqi},
  booktitle={IJCAI}, 
  year={2024},
}
```
