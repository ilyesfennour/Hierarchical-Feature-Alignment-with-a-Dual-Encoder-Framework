# Create task dataset
import torch
import numpy as np 
from PIL import Image
from pathlib import Path
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

class TaskDataset(Dataset):
    def __init__(self, dataset='ImageNet', root_dir='/data', split='train', attribute=None,
                 degradation=None, max_samples_per_class=None, sigma=75/255.0, scale_factor=4, **kwargs):

        self.degradation = degradation
        self.sigma = sigma
        self.scale_factor = scale_factor
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        if dataset == 'MedMNIST':
            from data import MedMNISTLoader
            self.data_loader = MedMNISTLoader(datasets=root_dir, split=split,
                                              max_samples_per_dataset=max_samples_per_class, **kwargs)
        elif dataset == 'ImageNet':
            from data import ImageNetLoader
            self.data_loader = ImageNetLoader(root_dir=root_dir, split=split,
                                              max_samples_per_class = max_samples_per_class, **kwargs)
        elif dataset == 'CelebA':
            from data import CelebALoader
            self.data_loader = CelebALoader(root_dir=root_dir, split=split, attribute=attribute,
                                              max_samples_per_class = max_samples_per_class, **kwargs)
        elif dataset == 'MNIST':
            from data import MNISTLoader
            self.data_loader = MNISTLoader(split=split, max_samples=max_samples_per_class, **kwargs)
        elif dataset == 'CIFAR10':
            from data import CIFAR10Loader
            self.data_loader = CIFAR10Loader(split=split, max_samples=max_samples_per_class, **kwargs)
        elif dataset == 'SVHN':
            from data import SVHNLoader
            self.data_loader = SVHNLoader(split=split, max_samples=max_samples_per_class, **kwargs)
        else:
            raise ValueError(f"Unknown dataset: '{dataset}'")
    
    def __len__(self):
        return len(self.data_loader[0])

    def __getitem__(self, idx ):
        img = self.data_loader[0][idx]

        if isinstance(img, (str, Path)):
            img = Image.open(img).convert('RGB')

        if isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                # If shape is (32, 32, 3) - HWC format
                if img.shape[-1] == 3:
                    img = Image.fromarray(img.astype('uint8'))
                # If shape is (3, 32, 32) - CHW format
                elif img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                    img = Image.fromarray(img.astype('uint8'))
                else:
                    raise ValueError(f"Unexpected image shape: {img.shape}")
            
            elif len(img.shape) == 2:
                # Grayscale (32, 32)
                img = Image.fromarray(img.astype('uint8')).convert('RGB')

            else:
                raise ValueError(f"Unexpected image dimensions: {img.shape}")
        
        elif isinstance(img, torch.Tensor):
            # Tensor, convert to PIL via numpy
            img_np = img.cpu().numpy()
            if len(img_np.shape) == 2:
                img = Image.fromarray(img_np).convert('RGB')
            else:
                img = Image.fromarray(img_np)

        img = self.transform(img)

        if self.degradation:
            import degradation as degr
            degraded, target = degr.apply_degradation(img, self.degradation, self.sigma, self.scale_factor)
            return {'degraded': degraded, 'clean': target}
        
        else:
            return {'degraded': img, 'clean': img, 'label': self.data_loader[1][idx]}

def get_data_loaders(batch_size=32, workers=0, degradation=None, **kwargs):
    """Create data loaders for TissueMNIST dataset"""
    class_weights_dict = None
    
    # Load dataset
    train_dataset = TaskDataset(split='train', degradation=degradation, **kwargs)
    val_dataset = TaskDataset(split='val', degradation=degradation, **kwargs)
    test_dataset = TaskDataset(split='test', degradation=degradation, **kwargs)

    num_classes = train_dataset.data_loader[2]

    if degradation is None:
        # Get class weights from training dataset
        train_labels = train_dataset.data_loader[1]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print("="*70)
        print("CLASS IMBALANCE ANALYSIS:")
        print()
        class_counts = Counter(train_labels)
        for class_idx in range(len(class_weights)):
            print(f"Class {class_idx}: {class_counts[class_idx]} samples, weight: {class_weights[class_idx]:.3f}")

        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        ratio = max_count // min_count

        print(f"\nRatio: {ratio:.1f} should be less than 100")
        print(f"Largest class: {max_count} samples")
        print(f"Smallest class: {min_count} samples")
        print(f"Number of classes: {num_classes} class")
        print("="*70)
    
    pin_memory = torch.cuda.is_available()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=True if workers > 0 else False,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, num_classes, class_weights_dict
