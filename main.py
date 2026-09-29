import os
import json
import torch
import multiprocessing

def run_training(autoencoder='normal', losses_config=None, dataset='MedMNIST', root_dir=None,
                 degradation=None, max_samples_per_class=None, epochs=50, batch_size=32, 
                 learning_rate=2e-3, workers=0, num_layers=4, pretrained_path=None,
                 checkpoint_dir=None, additional_epochs=0, sigma=75/255.0, scale_factor=4):
    """Run training with custom parameters"""

    multiprocessing.set_start_method('spawn', force=True)

    if checkpoint_dir:
        from checkpoint_utils import get_parameters
        autoencoder, losses_config, dataset, root_dir, degradation, max_samples_per_class, epochs, batch_size, learning_rate, workers, num_layers, pretrained_path = get_parameters(checkpoint_dir)
    
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
        model = LAE(num_layers=num_layers)
    elif autoencoder == 'DLAE':
        from model import DLAE
        model = DLAE(num_layers=num_layers, num_classes=num_classes)
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
                                       learning_rate=learning_rate, checkpoint_dir=checkpoint_dir, additional_epochs=additional_epochs,
                                       class_weights_dict=class_weights_dict, num_classes=num_classes)
    else:
        from training import reconstruction_train
        history = reconstruction_train(model, train_loader, val_loader, losses_config, device, exp_dir, num_epochs=epochs,
                                       learning_rate=learning_rate, checkpoint_dir=checkpoint_dir, additional_epochs=additional_epochs)

    print("=" * 70)
    print(f"TRAINING COMPLETED!")
    print(f"All results saved to: {exp_dir}")
    print("=" * 70)
    
    return history, exp_dir