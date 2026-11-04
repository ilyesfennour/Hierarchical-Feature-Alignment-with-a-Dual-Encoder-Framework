import os
import json
import torch
import argparse
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(description="Train autoencoder models.")

    # Configuration files
    parser.add_argument("--config", type=str, default=None, help="Path to the main JSON configuration file.")
    parser.add_argument("--losses_config", type=str, default=None, help="Path to the JSON losses configuration file.")

    # Model parameters
    parser.add_argument("--autoencoder", type=str, default=None,
        choices=["DFA", "LAE", "DLAE"]
    )
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--tap_layers", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default=None,
        choices=[
            "MedMNIST", "ImageNet", "MNIST", "CIFAR10",
            "CelebA", "SVHN", "DFB-HR"
        ]
    )
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--max_samples_per_class", type=int, default=None)
    parser.add_argument("--degradation", type=str, default=None,
        choices=["denoising", "super_resolution", "missing_data", "none"]
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)

    # Other parameters
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--scale_factor", type=int, default=None)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--additional_epochs", type=int, default=None)

    args = parser.parse_args()

    # Default parameters
    config = {
        "autoencoder": "DFA",
        "dataset": "MNIST",
        "root_dir": None,
        "degradation": 'denoising',
        "max_samples_per_class": None,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "workers": 0,
        "num_layers": 4,
        "tap_layers": 3,
        "latent_dim": 128,
        "sigma": 75 / 255.0,
        "scale_factor": 4,
        "pretrained_path": None,
        "checkpoint_dir": None,
        "additional_epochs": 0,
        "losses_config": {
            "l1": {"lambda": 1.0},
            "l2": {"lambda": 0.5},
            "perceptual": {"lambda": 0.1},
            "tap": {"lambda": 0.05}
        }
    }

    # Load main configuration file
    if args.config is not None:
        with open(args.config, "r") as f:
            file_config = json.load(f)
        config.update(file_config)

    # Override with explicitly provided command-line arguments
    cli_args = vars(args).copy()
    cli_args.pop("config")
    cli_args.pop("losses_config")

    for key, value in cli_args.items():
        if value is not None:
            config[key] = value

    # Load separate losses configuration if provided
    if args.losses_config is not None:
        with open(args.losses_config, "r") as f:
            config["losses_config"] = json.load(f)

    return argparse.Namespace(**config)

def run_training(autoencoder='DFA', losses_config=None, dataset='MNIST', root_dir=None,
                 degradation=None, max_samples_per_class=None, epochs=50, batch_size=32, 
                 learning_rate=2e-4, workers=0, num_layers=4, tap_layers=3, latent_dim=128,
                 pretrained_path=None, checkpoint_dir=None, additional_epochs=0, sigma=75/255.0, scale_factor=4):
    """Run training with custom parameters"""

    if tap_layers > num_layers - 1:
        raise ValueError(
            f"TAP_LAYERS ({tap_layers}) must be <= NUM_LAYERS - 1 ({num_layers - 1})."
        )

    multiprocessing.set_start_method('spawn', force=True)

    if checkpoint_dir:
        from checkpoint_utils import get_parameters
        autoencoder, losses_config, dataset, root_dir, degradation, max_samples_per_class, epochs, batch_size, learning_rate, workers, num_layers, tap_layers, latent_dim, pretrained_path = get_parameters(checkpoint_dir)
    
    if degradation:
        return_labels = False
    else:
        return_labels = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print("=" * 70)

    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Workers: {workers}")
    if pretrained_path:
        print(f"Pre-trained weights: {pretrained_path}")
    print("=" * 70)

    print(f"Autoencoder: {autoencoder}")
    print("Losses:", ', '.join(losses_config.keys()))
    print(f"Dataset: {dataset}")
    print(f"Root: {root_dir}")
    print(f"Degradation: {degradation}")
    print(f"Max samples per class: {max_samples_per_class}")
    print("=" * 70)
    
    # Get data loaders
    if dataset == 'DFB-HR':
        from get_patched_data_loaders import get_patched_data_loaders as data_loaders
    else:
        from get_data_loaders import get_data_loaders as data_loaders

    train_loader, val_loader, num_classes, class_weights_dict = data_loaders(
        dataset=dataset,
        root_dir=root_dir,
        degradation=degradation,
        max_samples_per_class=max_samples_per_class,
        return_labels=return_labels,
        batch_size=batch_size,
        workers=workers,
        sigma=sigma,
        scale_factor=scale_factor
    )
    
    # Create model
    if autoencoder == 'DFA':
        from model import DFA
        model = DFA(num_layers=num_layers)
    elif autoencoder == 'LAE':
        from model import LAE
        model = LAE(num_layers=num_layers, latent_dim=latent_dim)
    elif autoencoder == 'DLAE':
        from model import DLAE
        model = DLAE(num_layers=num_layers, latent_dim=latent_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown autoencoder: {autoencoder}")
    model.to(device)
    
    if pretrained_path:
        from checkpoint_utils import load_pretrained
        model = load_pretrained(model, pretrained_path, device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)
    
    # Create experiment directory
    exp_name = f"{dataset}_{degradation}_{autoencoder}_bs{batch_size}"
    counter = 1
    while True:
        exp_dir = f"training/{exp_name}_{counter}"
        if not os.path.exists(exp_dir):
            break
        counter += 1
    os.makedirs(exp_dir)
    print(f"Experiment directory: {exp_dir}")
    print()
    
    # Save configuration
    config = {
        'model': autoencoder,
        'losses_config': losses_config,
        'num_epochs': epochs,
        'dataset': dataset,
        'num_classes': num_classes,
        'max_samples_per_class': max_samples_per_class,
        'degradation': degradation,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'workers': workers,
        'num_layers': num_layers,
        'tap_layers': tap_layers,
        'latent_dim': latent_dim,
        'class_weights': class_weights_dict,
        'root_dir': root_dir,
        'pretrained_path': pretrained_path
    }
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Train model
    if return_labels:
        from training import classification_train
        history = classification_train(model, train_loader, val_loader, losses_config, device, exp_dir, num_epochs=epochs,
                                       tap_layers=tap_layers, learning_rate=learning_rate, checkpoint_dir=checkpoint_dir,
                                       additional_epochs=additional_epochs, class_weights_dict=class_weights_dict, num_classes=num_classes)
    else:
        from training import reconstruction_train
        history = reconstruction_train(model, train_loader, val_loader, losses_config, device, exp_dir, num_epochs=epochs, tap_layers=tap_layers,
                                       learning_rate=learning_rate, checkpoint_dir=checkpoint_dir, additional_epochs=additional_epochs)

    print("=" * 70)
    print(f"TRAINING COMPLETED!")
    print(f"All results saved to: {exp_dir}")
    print("=" * 70)
    
    return history, exp_dir

def main():
    args = parse_args()
    history, exp_dir = run_training(**vars(args))

    return history, exp_dir

if __name__ == "__main__":
    history, exp_dir = main()
