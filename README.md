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
| Motion blur      | Synthetic motion blur                             |
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
├── LICENSE
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
└── notebooks/
    └── training.ipynb
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
| `checkpoint_utils.py`         | Checkpoint loading, saving, and training continuation     |
| `notebooks/training.ipynb`    | Lightweight notebook interface for running experiments    |

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
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

Please download the datasets from their official sources and place them in the location specified by your configuration.

For example:

```text
data/
├── MNIST/
├── CIFAR10/
├── SVHN/
├── CelebA/
├── ImageNet/
├── MedMNIST/
└── DFB-HR/
```

The exact directory structure may depend on the dataset loader being used.

> **Note:** Dataset download instructions and preprocessing requirements should be added here for each dataset if they are not automatically handled by the corresponding loader.

---

# ▶️ Running an Experiment

The selected experiment determines:

1. The dataset
2. The degradation procedure
3. The appropriate data loader
4. The model configuration
5. The training procedure
6. The checkpoint configuration

---

# 🧠 Models

## DFA

The DFA model serves as a baseline autoencoder architecture for learning image representations and reconstructing degraded inputs.

---

## LAE

The LAE model extends the baseline architecture with a latent representation designed to provide a compact representation of the input.

---

## DLAE

DLAE extends the latent autoencoder with additional components for semantic representation learning and classification.

The latent representation can therefore be used for multiple objectives, including:

* Reconstruction
* Classification
* Feature alignment
* Semantic representation learning

---

# 📉 Loss Functions

The training framework supports several loss components.

The total training objective is constructed from weighted loss terms, including:

* L1 loss
* L2 loss
* Perceptual loss
* TAP loss
* Latent-space losses
* SSIM-based loss
* Edge loss
* Classification loss
* Center loss
* Batch-hard triplet loss

---

# 🖼️ Image Degradation

The degradation pipeline is implemented in:

```text
degradation.py
```

Available degradation operations include:

### Gaussian noise

```text
Clean image
     ↓
Gaussian noise
     ↓
Degraded image
     ↓
Autoencoder
     ↓
Reconstructed image
```

### Motion blur

```text
Clean image
     ↓
Motion blur
     ↓
Blurred image
     ↓
Autoencoder
     ↓
Reconstructed image
```

### Super-resolution

```text
High-resolution image
        ↓
     Downsample
        ↓
Low-resolution image
        ↓
    Autoencoder
        ↓
High-resolution reconstruction
```

### Missing data

```text
Original image
      ↓
Missing-data mask
      ↓
Degraded image
      ↓
Autoencoder
      ↓
Recovered image
```

---

# 🏋️ Training

Training procedures are implemented in:

```text
training.py
```

---

# 💾 Checkpoints

Checkpoint management is implemented in:

```text
checkpoint_utils.py
```

The training framework supports continuing training from an existing checkpoint and loading pretrained models.

---

# 📊 Evaluation

The framework evaluates the models using task-specific metrics.

### Image restoration

* PSNR
* SSIM

### Classification

* Accuracy

### Clustering

* Accuracy
* NMI
* ARI
* Davies–Bouldin Index

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
