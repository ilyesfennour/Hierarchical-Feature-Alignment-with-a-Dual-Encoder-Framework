# Model Evaluation
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    normalized_mutual_info_score,
    adjusted_rand_score,
    davies_bouldin_score
)
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

from model import DFA, LAE, DLAE
from get_data_loaders import TaskDataset

DATASET_PATHS = {
    "CBSD68": "test/dataset/train",
    "Kodak": "test/Kodak24/Kodak24",
    "McMaster": "test/McMaster",
    "Set12": "test/Set12",
    "Urban100": "test/Urban100/original",
    "BSD68": "test/BSD68",
    "Set5": "test/Set5/HR",
    "Set14": "test/Set14/original",
    "BSD100": "test/BSD100",
    "DIV2K": "test/DIV2K_valid_HR/DIV2K_valid_HR",
    "ParisStreetView": "test/paris_train_original",
    "Places2": "test/Places2/test_256",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation")

    # Configuration file
    parser.add_argument("--config", type=str, default=None, help="Path to the evaluation JSON configuration.")

    # Evaluation parameters
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None,
        choices=["restoration", "classification", "clustering"]
    )

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default=None,
        choices=[
            "CelebA", "SVHN", "CBSD68", "Kodak", "McMaster",
            "Set12", "Urban100", "BSD68", "Set5", "Set14",
            "BSD100", "DIV2K", "ParisStreetView", "Places2"
        ]
    )
    parser.add_argument("--root_dir", type=str, default=None)

    # Degradation parameters
    parser.add_argument("--degradation", type=str, default=None,
        choices=["denoising", "super_resolution", "missing_data", "none"]
    )
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--scale_factor", type=int, default=None)

    # DataLoader parameters
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)

    # Default parameters
    config = {
        "checkpoint": None,
        "mode": "restoration",
        "dataset": "CelebA",
        "root_dir": None,
        "degradation": "denoising",
        "sigma": 75 / 255.0,
        "scale_factor": 4,
        "batch_size": 32,
        "workers": 0
    }

    # Load configuration file
    if args.config is not None:
        with open(args.config, "r") as f:
            file_config = json.load(f)
        config.update(file_config)

    # Override with explicitly provided command-line arguments
    cli_args = vars(args).copy()
    cli_args.pop("config")

    for key, value in cli_args.items():
        if value is not None:
            config[key] = value

    return argparse.Namespace(**config)

def evaluate_model(model, test_loader, device):

    psnr_values = []
    ssim_values = []

    model.eval()

    print("\nEvaluating model...")

    with torch.no_grad():
        for batch in tqdm(test_loader):

            degraded = batch["degraded"].to(device)
            clean = batch["clean"].to(device)

            output, _ = model(degraded)

            # Evaluate each image independently
            for i in range(clean.size(0)):

                clean_np = clean[i].detach().cpu().permute(1, 2, 0).numpy()
                output_np = output[i].detach().cpu().permute(1, 2, 0).numpy()

                # Handle grayscale images
                if clean_np.shape[-1] == 1:
                    clean_np = clean_np.squeeze(-1)
                    output_np = output_np.squeeze(-1)
                    channel_axis = None
                else:
                    channel_axis = -1

                psnr_values.append(psnr(clean_np, output_np, data_range=1.0))
                ssim_values.append(ssim(clean_np, output_np, data_range=1.0, channel_axis=channel_axis))

    if not psnr_values:
        raise RuntimeError("The evaluation dataset is empty.")

    stats = {
        "num_images": len(psnr_values),
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values)),
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_std": float(np.std(ssim_values))
    }

    print("\nEvaluation Results")
    print(f"Number of images : {stats['num_images']}")
    print(f"PSNR             : {stats['psnr_mean']:.4f} ± {stats['psnr_std']:.4f}")
    print(f"SSIM             : {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")

    return stats

def evaluate_classification(model, test_loader, device):

    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating classification"):

            images = batch["degraded"].to(device)
            labels = batch["label"].to(device).long()

            # Forward pass
            _, _, logits = model(images)

            # Predicted classes
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise RuntimeError("The evaluation dataset is empty.")

    accuracy = 100.0 * correct / total

    print(f"\nClassification Accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{total}")

    return accuracy

def extract_latents(model, dataloader, device):
    """Extract latent representations and ground-truth labels."""
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latent features"):
            images = batch["degraded"].to(device)
            targets = batch["label"].cpu().numpy().reshape(-1)

            _, z, _ = model(images)

            latents.append(z.reshape(z.size(0), -1).cpu().numpy())
            labels.append(targets)

    return np.concatenate(latents, axis=0), np.concatenate(labels, axis=0)

def clustering_accuracy(y_true, y_pred):
    """Calculate clustering accuracy using the Hungarian algorithm."""
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / cm.sum()

def evaluate_clustering(model, test_loader, device, n_clusters=None):
    """Evaluate latent-space clustering using K-Means and four metrics."""

    # 1. Extract latent representations
    latents, y_true = extract_latents(model, test_loader, device)

    # 2. Normalize latent representations
    latents = StandardScaler().fit_transform(latents)

    # 3. Determine the number of clusters
    if n_clusters is None:
        n_clusters = len(np.unique(y_true))

    # 4. Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    y_pred = kmeans.fit_predict(latents)

    # 5. Calculate clustering metrics
    acc = clustering_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    dbi = davies_bouldin_score(latents, y_pred)

    results = {
        "ACC": acc,
        "NMI": nmi,
        "DBI": dbi,
        "ARI": ari
    }

    print("\nClustering Evaluation")
    print("-" * 30)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return results

# Load Model
def load_model(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint["name"]
    model_classes = {"DFA": DFA, "LAE": LAE, "DLAE": DLAE}

    if model_name not in model_classes:
        raise ValueError(f"Unknown model architecture: {model_name}")

    model_class = model_classes[model_name]

    # Common parameters
    model_args = {"num_layers": checkpoint["num_layers"]}

    # LAE and DLAE require latent_dim
    if model_name in ["LAE", "DLAE"]:
        model_args["latent_dim"] = checkpoint["latent_dim"]

    # DLAE requires num_classes
    if model_name == "DLAE":
        model_args["num_classes"] = checkpoint["num_classes"]

    model = model_class(**model_args)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model: {model_name}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Number of layers: {checkpoint['num_layers']}")

    if checkpoint["latent_dim"] is not None:
        print(f"Latent dimension: {checkpoint['latent_dim']}")

    if checkpoint["num_classes"] is not None:
        print(f"Number of classes: {checkpoint['num_classes']}")

    return model

# Create Evaluation DataLoader
def get_dataset_config(dataset, root_dir):
    if dataset in ["CelebA", "SVHN"]:
        return dataset, root_dir

    if dataset in DATASET_PATHS:
        dataset_path = os.path.join(root_dir, DATASET_PATHS[dataset])

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_path}"
            )

        return "Test", dataset_path

    raise ValueError(f"Unknown dataset: {dataset}")

def create_test_loader(args):
    dataset_name, dataset_root = get_dataset_config(args.dataset, args.root_dir)

    test_dataset = TaskDataset(
        dataset=dataset_name,
        root_dir=dataset_root,
        split="test",
        degradation=None if args.degradation == "none" else args.degradation,
        sigma=args.sigma,
        scale_factor=args.scale_factor
    )

    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.checkpoint, device)

    # Create test loader
    test_loader = create_test_loader(args)

    print(f"\nDataset: {args.dataset}")
    print(f"Evaluation mode: {args.mode}")
    print(f"Device: {device}")

    if args.mode == "restoration":
        if args.degradation is None or args.degradation == "none":
            raise ValueError("Restoration requires a degradation.")

        return evaluate_model(model, test_loader, device)

    elif args.mode == "classification":
        if not isinstance(model, DLAE):
            raise ValueError("Classification requires DLAE.")

        return evaluate_classification(model, test_loader, device)

    elif args.mode == "clustering":
        if not isinstance(model, DLAE):
            raise ValueError("Clustering requires DLAE.")

        return evaluate_clustering(model, test_loader, device)

if __name__ == "__main__":
    args = parse_args()
    results = main(args)
