import os
import json
import torch
import argparse
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder Training")

    # Model configuration
    parser.add_argument("--autoencoder", type=str, default="DFA",
        choices=["DFA", "LAE", "DLAE"]
    )
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--tap_layers", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128)

    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="MNIST",
        choices=[
            "MedMNIST", "ImageNet", "MNIST", "CIFAR10",
            "CelebA", "SVHN", "DFB-HR"
        ]
    )
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--max_samples_per_class", type=int, default=None)

    # Degradation
    parser.add_argument("--degradation", type=str, default="denoising",
        choices=["denoising", "super_resolution", "missing_data", "none"]
    )
    parser.add_argument("--sigma", type=float, default=75/255.0)
    parser.add_argument("--scale_factor", type=int, default=4)

    # Training configuration
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=0)

    # Checkpoints
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--additional_epochs", type=int, default=0)

    # Loss configuration (JSON file)
    parser.add_argument("--losses_config", type=str, default=None)

    return parser.parse_args()

def run_training(autoencoder='DFA', losses_config=None, dataset='MNIST', root_dir=None,
                 degradation=None, max_samples_per_class=None, epochs=50, batch_size=32, 
                 learning_rate=2e-4, workers=0, num_layers=4, tap_layers=3, latent_dim=128,
                 pretrained_path=None, checkpoint_dir=None, additional_epochs=0, sigma=75/255.0, scale_factor=4):
    """Run training with custom parameters"""

    if TAP_LAYERS > NUM_LAYERS - 1:
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

def main(args):
    # Load loss configuration
    if args.losses_config:
        with open(args.losses_config, "r") as f:
            losses_config = json.load(f)
    else:
        losses_config = {
            "l1": {"lambda": 1.0},
            "l2": {"lambda": 0.5},
            "perceptual": {"lambda": 0.1},
            "tap": {"lambda": 0.05}
        }

    return run_training(
        autoencoder=args.autoencoder,
        losses_config=losses_config,
        dataset=args.dataset,
        root_dir=args.root_dir,
        degradation=None if args.degradation == "none" else args.degradation,
        max_samples_per_class=args.max_samples_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        workers=args.workers,
        num_layers=args.num_layers,
        tap_layers=args.tap_layers,
        latent_dim=args.latent_dim,
        pretrained_path=args.pretrained_path,
        checkpoint_dir=args.checkpoint_dir,
        additional_epochs=args.additional_epochs,
        sigma=args.sigma,
        scale_factor=args.scale_factor
    )

if __name__ == "__main__":
    args = parse_args()
    history, exp_dir = main(args)
