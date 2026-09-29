# Create task dataset
import torch
import random
import numpy as np 
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight


class TaskDataset(Dataset):
    def __init__(self, dataset=None, root_dir=None, split='train', patch_size=64, stride=32,
                 degradation=None, max_samples_per_class=None, cache_size=10000, **kwargs):
        self.patch_size = patch_size
        self.stride = stride
        self.split = split
        self.degradation = degradation
        self.images = []
        self.tensor = transforms.Compose([transforms.ToTensor()])
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

        if dataset == 'DFB-HR':
            from data import DFBLoader
            self.data_loader = DFBLoader(root_dir, split, max_samples_per_class)

        if split == 'train':
            self.patch_metadata = []
            max_patches = 0

            for img_idx in tqdm(range(len(self.data_loader[0])), desc='Extracting pathes'):
                img_path = self.data_loader[0][img_idx]
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)

                patches = F.unfold(img_tensor.unsqueeze(0), kernel_size=patch_size, stride=stride)
                num_patches = patches.shape[2]

                if num_patches > max_patches:
                    max_patches = num_patches
                for patch_idx in range(num_patches):
                    for aug_idx in range(9):
                        self.patch_metadata.append({
                            'img_idx': img_idx,
                            'patch_idx': patch_idx,
                            'aug_idx': aug_idx,
                            'img_path': img_path
                        })

            self.total = len(self.patch_metadata)
            if (max_patches*9) > self.cache_size:
                raise ValueError(f"Cache size({self.cache_size}) should be greater than {(max_patches*3)}")

            print(f"Created {self.total} samples ({len(self.data_loader[0])} images → {self.total} augmented patches)")
            print(f"Max number of patches: {max_patches}")

        else:
            for img_idx in range(len(self.data_loader[0])):
                img = Image.open(self.data_loader[0][img_idx]).convert('RGB')
                img = self.transform(img)
                self.images.append(img)
            self.total = len(self.images)

    def __len__(self):
        return self.total

    def augment_patch(self, patch, aug_idx):
        """Apply 2 different augmentations"""
        if aug_idx == 0:  # Original
            return patch
        elif aug_idx == 1:  # Horizontal flip
            return torch.flip(patch, dims=[2])
        elif aug_idx == 2:  # Vertical flip
            return torch.flip(patch, dims=[1])
        elif aug_idx == 3:  # Rotate 90 degrees
            return torch.rot90(patch, k=1, dims=[1, 2])
        elif aug_idx == 4:  # Rotate 180 degrees
            return torch.rot90(patch, k=2, dims=[1, 2])
        elif aug_idx == 5:  # Small random rotation (±15 degrees)
            angle = random.uniform(-15, 15)
            return transforms.functional.rotate(patch, angle)
        elif aug_idx == 6:  # Random brightness adjustment
            brightness = random.uniform(0.7, 1.3)
            return patch * brightness
        elif aug_idx == 7:  # Random contrast adjustment
            mean = patch.mean(dim=[1,2], keepdim=True)
            contrast = random.uniform(0.7, 1.3)
            return (patch - mean) * contrast + mean
        elif aug_idx == 8:  # Random scaling (zoom)
            scale = random.uniform(0.8, 1.2)
            h, w = patch.shape[1], patch.shape[2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = transforms.functional.resize(patch, [new_h, new_w])
            # Center crop back to original size
            crop_h = (new_h - h) // 2 if new_h > h else 0
            crop_w = (new_w - w) // 2 if new_w > w else 0
            if new_h > h and new_w > w:
                scaled = scaled[:, crop_h:crop_h+h, crop_w:crop_w+w]
            else:
                scaled = transforms.functional.resize(scaled, [h, w])
            return scaled

    def cache_image(self, idx):
        """Load image and cache ALL its patches"""
        if idx in self.cache:
            return self.cache[idx]

        # Load and process image
        meta = self.patch_metadata[idx]
        img = Image.open(meta['img_path']).convert('RGB')
        img_tensor = self.transform(img)

        patches = img_tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
        patches = patches.contiguous().view(3, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1, 0, 2, 3)

        count = 0
        for patch in patches:
            for aug_idx in range(9):
                # Remove oldest if cache full
                if len(self.cache) >= self.cache_size:
                    oldest = self.cache_order.pop(0)
                    del self.cache[oldest]

                # Store in cache
                self.cache[count + idx] = self.augment_patch(patch, aug_idx)
                self.cache_order.append(count + idx)
                count += 1

        return self.cache[idx]

    def __getitem__(self, idx):
        if self.split == 'train':
            img = self.cache_image(idx)
        else:
            img = self.images[idx]

        if self.degradation:
            import degradation as degr
            degraded, target, _ = degr.apply_degradation(img, self.degradation)

        return {'degraded': degraded, 'clean': target}

def get_patched_data_loaders(batch_size=32, workers=0, degradation=None, **kwargs):
    """Create data loaders for TissueMNIST dataset"""
    class_weights_dict = None
    
    # Load TissueMNIST dataset
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, num_classes, class_weights_dict