# Hierarchical Feature Alignment with a Dual-Encoder Framework for Unified Image Restoration, Classification, Visualization, and Clustering

Official implementation of the experiments presented in:

> **Multi-task Autoencoders for Image Denoising, Enhancement, Clustering, Classification, and Data Recovery**

This repository provides the training and evaluation code for a family of autoencoder-based models designed to address multiple image-processing tasks within a unified framework.

The implementation includes:

* Image denoising
* Image super-resolution
* Image inpainting / missing-data recovery
* Image classification
* Image clustering
* Latent-space representation learning
* Transfer/alignment of semantic information through the latent representation

---

## 📌 Overview

The proposed framework investigates autoencoder architectures for learning useful image representations while simultaneously supporting different downstream tasks.

The repository contains implementations of three main models:

* **DFA** — baseline dual-feature autoencoder
* **LAE** — latent autoencoder
* **DLAE** — dual latent autoencoder with an additional prediction/classification component

The models can be trained with different datasets and image degradation types.

### Supported degradation types

| Degradation      | Description                                       |
| ---------------- | ------------------------------------------------- |
| Gaussian noise   | Additive Gaussian noise                           |
| Super-resolution | Downsampling followed by reconstruction           |
| Missing data     | Removal of image information followed by recovery |

### Supported datasets

The current implementation includes loaders for:

* MNIST
* CIFAR-10
* SVHN
* CelebA
* ImageNet
* MedMNIST
* DFB-HR

The DFB-HR dataset is additionally supported through a patch-based data-loading pipeline for high-resolution image restoration experiments.

---

## 🏗️ Repository Structure

```text
.
├── README.md
├── requirements.txt
│
├── main.py
├── model.py
├── data.py
├── degradation.py
├── get_data_loaders.py
├── get_patched_data_loaders.py
├── training.py
├── training_utilities.py
├── checkpoint_utils.py
│
├── configs/
│   ├── training/
│   ├── evaluation/
│   └── losses/
│
└── notbock/
    └── experiments.ipynb
```

### Main files

| File                          | Description                                               |
| ----------------------------- | --------------------------------------------------------- |
| `main.py`                     | Main entry point and experiment configuration             |
| `model.py`                    | Definitions of DFA, LAE, DLAE, and model components       |
| `data.py`                     | Dataset loading and dataset-specific preprocessing        |
| `degradation.py`              | Image degradation functions                               |
| `get_data_loaders.py`         | Construction of standard training/validation/test loaders |
| `get_patched_data_loaders.py` | Patch-based data loading for DFB-HR                       |
| `training.py`                 | Reconstruction and classification training procedures     |
| `training_utilities.py`       | Loss functions and training utilities                     |
| `checkpoint_utils.py`         | Checkpoint loading, and training continuation             |
| `evaluate.py`                 | Evaluation procedures and performance metrics             |
| `notebooks/training.ipynb`    | Lightweight notebook interface for running experiments    |

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/ilyesfennour/Hierarchical-Feature-Alignment-with-a-Dual-Encoder-Framework.git
cd Hierarchical-Feature-Alignment-with-a-Dual-Encoder-Framework
```

## 2. Create a virtual environment

We recommend using a dedicated Python environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training, install the appropriate PyTorch version for your CUDA configuration if necessary.

---

# 📂 Dataset Preparation

The repository does **not** include the datasets.

### Automatic Download

No manual download is required for the following datasets, which are downloaded automatically by their respective loaders:

- **MNIST**
- **MedMNIST**
- **CIFAR-10**
- **SVHN**

### Manual Download — Training Datasets

Download the following datasets from Kaggle:

- **CelebA:** [Kaggle Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- **DFB-HR:** Combine the following three sources into a single `DFB-HR/` folder:
  - DIV2K: [Kaggle Dataset](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images)
  - Flickr2K: [Kaggle Dataset](https://www.kaggle.com/datasets/daehoyang/flickr2k)
  - BSD500: [Kaggle Dataset](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500)
- **ImageNet:** Merge the following four sources into `train/`:
  - [ImageNet Part 0](https://www.kaggle.com/datasets/sautkin/imagenet1k0)
  - [ImageNet Part 1](https://www.kaggle.com/datasets/sautkin/imagenet1k1)
  - [ImageNet Part 2](https://www.kaggle.com/datasets/sautkin/imagenet1k2)
  - [ImageNet Part 3](https://www.kaggle.com/datasets/sautkin/imagenet1k3)

  Download the [ImageNet validation dataset](https://www.kaggle.com/datasets/sautkin/imagenet1kvalid) and place it in `test/`, as a sibling of `train/`.

### Manual Download — Evaluation Benchmarks

The following datasets are used for evaluating image restoration, denoising, and super-resolution performance:

**Image Inpainting / Completion**
- **CelebA:** [Kaggle Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- **SVHN:** [Kaggle Dataset](https://www.kaggle.com/datasets/xiake666/svhm-dataset)
- **Paris StreetView:** [Kaggle Dataset](https://www.kaggle.com/datasets/arrowjrssgss/parisstreet)
- **Places2:** [Kaggle Dataset](https://www.kaggle.com/datasets/nguynhoitrung/places2-256-flatten)

**Image Denoising**
- **CBSD68:** [Kaggle Dataset](https://www.kaggle.com/datasets/tarekmebrouk/cbsd68)
- **Kodak24:** [Kaggle Dataset](https://www.kaggle.com/datasets/mustafaalkhafaji95/kodak-dataset)
- **McMaster:** [Kaggle Dataset](https://www.kaggle.com/datasets/liikke/mcmaster)
- **Set12:** [Kaggle Dataset](https://www.kaggle.com/datasets/abdulfatahnasrat/set12-test)
- **Urban100:** [Kaggle Dataset](https://www.kaggle.com/datasets/sahilchawla7/urban100)
- **BSD68:** [Kaggle Dataset](https://www.kaggle.com/datasets/zaminahmad/bsd68-dataset)

**Image Super-Resolution**
- **Set5:** [Kaggle Dataset](https://www.kaggle.com/datasets/chenqizhou/set5-hr-lr)
- **Set14:** [Kaggle Dataset](https://www.kaggle.com/datasets/sahilnsut/set14dataset)
- **BSD100:** [Kaggle Dataset](https://www.kaggle.com/datasets/huaiyingu/bsd100)
- **Urban100:** [Kaggle Dataset](https://www.kaggle.com/datasets/sahilchawla7/urban100)
- **DIV2K:** [Kaggle Dataset](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images)

### Expected Directory Structure

```text
data/
├── CelebA/
├── DFB-HR/
│   ├── DIV2K/
│   ├── Flickr2K/
│   └── BSD500/
├── ImageNet/
│   ├── train/
│   └── test/
└── benchmarks/
    ├── ParisStreetView/
    ├── Places2/
    ├── CBSD68/
    ├── Kodak24/
    ├── McMaster/
    ├── Set12/
    ├── BSD68/
    ├── Urban100/
    ├── Set5/
    ├── Set14/
    └── BSD100/
```

---

# 🏋️ Training

The training pipeline is implemented in main.py. Models can be trained using either command-line arguments or JSON configuration files.

You can specify the training parameters directly from the command line.

Example: Training a DFA model on MNIST for image denoising

```bash
python main.py \
    --autoencoder DFA \
    --dataset MNIST \
    --degradation denoising \
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 0.0002 \
    --workers 4 \
    --num_layers 4 \
    --tap_layers 3 \
    --sigma 0.2941
```
For classification and clustering, use --degradation none

For convenience and reproducibility, the repository provides several example JSON configuration files in the configs/training/ directory.

These examples cover different model architectures, datasets, and training tasks. Users can select an existing configuration or modify its parameters to define their own experiments.

To run an experiment using a configuration file:

```bash
python main.py --config configs/training/mnist_dfa.json
```

The repository provides four example loss configuration files in configs/losses/, corresponding to the supported model architectures and variants:

dfa_losses.json — Loss configuration for DFA.

lae_losses.json — Loss configuration for LAE.

rc_loss.json — Loss configuration for the RC variant.

mr_loss.json — Loss configuration for the MR variant.

To use a loss configuration, pass its path through --losses_config. For example:

```bash
python main.py \
    --config configs/training/mnist_dfa.json \
    --losses_config configs/losses/dfa_losses.json
```

Select the appropriate loss file for your model and experiment. You can also modify the provided configurations to adjust loss functions and their weights without changing the training code.

Individual CLI arguments can override values specified in a JSON configuration file.

For example:
```bash
python main.py \
    --config configs/training/mnist_dfa.json \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0001
```
The configuration priority is:
1. Default parameters.
2. JSON configuration file.
3. Explicit command-line arguments.
The separate --losses_config argument can also be used to specify or replace the loss configuration.
---

# 💾 Training continuation
The training pipeline supports:
- Saving model checkpoints.
- Loading pretrained weights.
- Resuming training from existing checkpoints.
- Continuing training for additional epochs.
The evaluation pipeline loads saved checkpoints and reports the corresponding quantitative metrics.

Checkpoint directories and experiment configurations can be specified through the training and evaluation arguments.
Training can be continued from an existing checkpoint by specifying its directory and the number of additional epochs.
```bash
python main.py \
    --checkpoint_dir checkpoints/experiment_1 \
    --additional_epochs 50
```
The checkpoint configuration is used to restore the previous experiment settings. Ensure that the checkpoint directory points to a valid saved experiment.

---

# 📓 Notebook

A lightweight Jupyter notebook is provided in:

```text
notebooks/training.ipynb
```

The notebook is intended as a convenient interface for launching experiments rather than containing the complete implementation.

The core implementation remains in the Python modules so that experiments can be reproduced from the command line and the code can be reused programmatically.

---

# 📝 Citation

If you use this code or the proposed models in your research, please cite the associated paper:

The citation information will be updated once the paper is published.

---

# 📜 License

This project is released under the license specified in:

---

# 🙏 Acknowledgements

We acknowledge the open-source datasets, libraries, and research projects that made this work possible.

In particular, this project makes use of the Python scientific-computing and deep-learning ecosystem, including PyTorch, torchvision, scikit-learn, and related libraries.

---

# 📬 Contact

For questions regarding the implementation or the research, please open an issue in this repository or contact the authors through their institutional/research profiles.
