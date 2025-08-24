# GPT From Scratch

This project demonstrates how to tokenize a dataset and train a
GPT-2 style model **from scratch**. Default parameters have been tested to work with a ASUS Zephyrus G14 with RTX 4060 GPU (8GB VRAM), 16 core CPU and 32 GB RAM.

------------------------------------------------------------------------

## Installation

```bash
conda create -n gpt-scratch python=3.11 -y
conda activate gpt-scratch
pip install torch torchvision torchaudio  # update for your CUDA version
pip install transformers datasets accelerate
pip install tensorboard tqdm pyarrow
```

------------------------------------------------------------------------

## 1. Tokenization

``` bash
python tokenize.py
```

Parameters to adjust

-   `num_proc` → set close to the number of **CPU cores** on your
    system.\
-   `batch_size` → increase as much as your **RAM capacity** allows.

------------------------------------------------------------------------

## 2. Pre-Training

``` bash
python pretrain.py
```

Parameters to adjust

-   `BATCH_SIZE` → depends on **GPU VRAM capacity**


## 3. Fine-tuning
Coming up soon