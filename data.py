import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

def CelebALoader(root_dir='/CelebA', split='train', attribute='Male', max_samples_per_class=None, return_labels=None):
    """ CelebA data loader (only load file paths) """
    
    # Set paths
    img_dir = Path(root_dir) / 'img_align_celeba/img_align_celeba'
    partition_csv = Path(root_dir) / 'list_eval_partition.csv'
    attr_csv = Path(root_dir) / 'list_attr_celeba.csv'

    if not img_dir.exists():
        raise ValueError(f"Image directory not found: {img_dir}")
    if not partition_csv.exists():
        raise ValueError(f"Partition CSV not found: {partition_csv}")
    if not attr_csv.exists():
        raise ValueError(f"Attributes CSV not found: {attr_csv}")

    if attribute is None:
        attribute = 'Male'

    # Load partition data
    partition_df = pd.read_csv(partition_csv)

    # Map split name to partition value
    split_map = {'train': 0, 'val': 1, 'test': 2}
    if split not in split_map:
        raise ValueError(f"Split must be one of {list(split_map.keys())}")

    partition_value = split_map[split]

    # Get images for the specified split
    split_images = partition_df[partition_df['partition'] == partition_value]['image_id'].values

    # Create label dictionary if attribute is specified
    label_dict = None
    attr_names = None

    if attribute is not None:
        # Load attributes
        attr_df = pd.read_csv(attr_csv)

        # Get attribute names (first column is image_id, rest are attributes)
        attr_names = attr_df.columns.tolist()[1:]

        # Verify attribute exists
        if attribute not in attr_names:
            raise ValueError(f"Attribute '{attribute}' not found. Available: {attr_names[:10]}...")

        # Create fast lookup dictionary {image_id: label}
        # Convert -1/1 to 0/1
        label_dict = dict(zip(attr_df['image_id'], (attr_df[attribute] == 1).astype(int)))

    # Prepare image paths and labels
    image_paths = []
    image_labels = [] if attribute is not None else None

    # For per-class sampling
    label_counts = {0: 0, 1: 0} if max_samples_per_class is not None else None

    for img_id in split_images:
        # Skip if we've reached max samples for both classes (when sampling is enabled)
        if max_samples_per_class is not None:
            if label_counts[0] >= max_samples_per_class and label_counts[1] >= max_samples_per_class:
                break

        # Get image path
        img_path = img_dir / img_id
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        if attribute is not None:
            # Get label from dictionary
            label = label_dict.get(img_id)
            if label is None:
                print(f"Warning: No label found for {img_id}, skipping...")
                continue

            # Check per-class limit
            if max_samples_per_class is not None:
                if label_counts[label] >= max_samples_per_class:
                    continue
                label_counts[label] += 1

            image_labels.append(label)

        image_paths.append(img_path)

    num_classes = len(np.unique(image_labels)) if attribute is not None and image_labels else None

    print(f"CelebA {split}" + (f" (attribute: {attribute})" if attribute else "") + f": {len(image_paths)} samples")
    if attribute is not None and image_labels:
        print(f"    Classes: {num_classes} classes")

    return image_paths, image_labels, num_classes

def ImageNetLoader(root_dir='/ImageNet', split='train', max_samples_per_class=None, return_labels=False):
    """ImageNet data loader"""

    split_dir = Path(root_dir)

    if split == 'test':
        split_dir = split_dir / 'test'
    else:
        split_dir = split_dir / 'train'

    if not split_dir.exists():
        raise ValueError(f"ImageNet directory not found: {split_dir}")

    # Initialize containers
    image_paths = []
    image_labels = [] if return_labels else None
    identity_to_label = {} if return_labels else None
    current_label = 0 if return_labels else None

    # Get all identity folders
    folders = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    print(f"Scanning {len(folders)} identities...")

    for identity_folder in folders:
        image_files = list(identity_folder.glob("*.jpg"))[:max_samples_per_class]
        max_samples = len(image_files)

        if split == 'train':
            image_files = image_files[:int(max_samples*0.85)]
        elif split == 'val':
            image_files = image_files[int(max_samples*0.85):]

        image_paths.extend(image_files)

        if return_labels and len(image_paths) != 0:
            identity_id = identity_folder.name
            if identity_id not in identity_to_label:
                identity_to_label[identity_id] = current_label
                current_label += 1
            label = identity_to_label[identity_id]
            image_labels.extend([label] * len(image_files))

    num_classes = len(np.unique(image_labels)) if return_labels else None

    print(f"ImageNet {split}: {len(image_paths)} samples")
    if return_labels:
        print(f"    Classes: {num_classes} classe")

    return image_paths, image_labels, num_classes

def DFBLoader(root_dir='trian', split='val', max_samples_per_class=None):
    """some dataset loader with lazy loading(only load file paths)"""

    # Initialize containers
    image_paths = []

    if split == 'train':
        dataset_path = ['DIV2K_train_HR/DIV2K_train_HR','Flickr2K','images/train']
    elif split == 'val':
        dataset_path = ['DIV2K_valid_HR/DIV2K_valid_HR','Flickr2K','images/val']
    elif split == 'test':
        dataset_path = ['images/test']

    for path in dataset_path:
        path = Path(path)
        split_dir = root_dir / path

        if not split_dir.exists():
            raise ValueError(f"directory not found: {split_dir}")

        image_files = list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpg"))[:max_samples_per_class]

        if path == 'Flickr2K':
            max_samples = int(len(image_files) * 0.85)

            if split == 'train':
                image_files = image_files[:max_samples]
            else:
                image_files = image_files[max_samples:]

        image_paths.extend(image_files)

    print(f"{split} dataset: {len(image_paths)} samples")

    return image_paths, None, None

def MedMNISTLoader(datasets=None, split='train', max_samples_per_dataset=None, return_labels=False):
    """MedMNIST data loader supporting all 2D datasets and multi-dataset combinations"""
    from medmnist import (
        TissueMNIST, BloodMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST,
        DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST,
        PathMNIST, ChestMNIST
    )

    DATASET_MAP = {
        'tissuemnist': TissueMNIST, 
        'bloodmnist': BloodMNIST,
        'organamnist': OrganAMNIST,
        'organcmnist': OrganCMNIST,
        'organsmnist': OrganSMNIST,
        'dermamnist': DermaMNIST,
        'octmnist': OCTMNIST,
        'pneumoniamnist': PneumoniaMNIST,
        'retinamnist': RetinaMNIST,
        'breastmnist': BreastMNIST,
        'pathmnist': PathMNIST,
        'chestmnist': ChestMNIST,
    }

    if datasets is None:
        # Return all available datasets
        datasets_to_load = [name for name, _ in DATASET_MAP.items() ]
    else:
        datasets_to_load = datasets

    images = []
    labels = []
    dataset_components = []

    current_global_offset = 0

    for dataset_name in datasets_to_load:
        if dataset_name not in DATASET_MAP:
            available = list(DATASET_MAP.keys())
            print(f"Warning: Dataset '{dataset_name}' not found. Available: {available}")
            continue

        dataset_class = DATASET_MAP[dataset_name]

        try:
            print(f"Loading {dataset_name}...")
            dataset = dataset_class(split=split, download=True, as_rgb=True)

            # Get images
            start_idx = len(images)
            images.extend(dataset.imgs[:max_samples_per_dataset])

            if return_labels:
                if dataset_name == 'chestmnist':
                    # ChestMNIST: labels are (N, 14) multi-hot
                    # Take pneumonia column (index 1) and flatten
                    dataset_labels = dataset.labels[:, 1].flatten()[:max_samples_per_dataset]
                else:
                    # All other datasets: flatten 1D array
                    dataset_labels = dataset.labels.flatten()[:max_samples_per_dataset]
            else:
                dataset_labels = None
            
            # Store dataset component info
            component_info = {
                'name': dataset_name,
            }

            # Adjust labels to be globally unique
            if return_labels and dataset_labels is not None:
                # Find unique labels in this dataset
                unique_labels = np.unique(dataset_labels)
                label_offset = current_global_offset

                # Store offset for this dataset
                class_info = {
                    'offset': label_offset,
                }
                component_info.update(class_info)

                # Add adjusted labels
                labels.extend((dataset_labels + label_offset).tolist())

                # Update global offset for next dataset
                current_global_offset += len(unique_labels)
            
            dataset_components.append(component_info)

            print(f"  ✓ {dataset_name}: {len(images) - start_idx} samples")
            if return_labels and dataset_labels is not None:
                print(f"    Classes: {len(np.unique(dataset_labels))} (global offset: {component_info['offset']})")

        except Exception as e:
            print(f"  ✗ Failed to load {dataset_name}: {e}")

    num_classes = len(np.unique(labels))

    print(f"MedMNIST {split}: {len(images)} total samples")
    print(f"Loaded datasets: {', '.join([d['name'] for d in dataset_components])}")
    print()

    return images, labels, num_classes

def MNISTLoader(split='train', max_samples=None, return_labels=False):
    from torchvision import datasets, transforms

    labels = None
    num_classes = None

    # Define a transform to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training set
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()

    # Download and load the test set
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    if split == 'test':
        images = x_test[:max_samples]
        if return_labels:
            labels = y_test[:max_samples]
            num_classes = len(np.unique(labels))
    elif split == 'train':
        images = x_train[:max_samples]
        images = images[:int(len(images) * 0.85)]
        if return_labels:
            labels = y_train[:len(images)]
            num_classes = len(np.unique(labels))
    else:
        images = x_train[:max_samples]
        images = images[int(len(images) * 0.85):]
        if return_labels:
            labels = y_train[:max_samples]
            labels = labels[int(len(labels) * 0.85):]
            num_classes = len(np.unique(labels))

    # Check the data shape
    print(f"{split} data: {len(images)}")
    if return_labels:
        print(f"Number of classes: {num_classes}")

    return images, labels, num_classes

def CIFAR10Loader(split='train', max_samples=None, return_labels=False):
    from torchvision import datasets, transforms

    labels = None
    num_classes = None

    # Define a transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training set
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    x_train = train_dataset.data
    y_train = train_dataset.targets

    # Download and load the test set
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    x_test = test_dataset.data
    y_test = test_dataset.targets

    if split == 'test':
        images = x_test[:max_samples]
        if return_labels:
            labels = y_test[:max_samples]
            num_classes = len(np.unique(labels))
    elif split == 'train':
        images = x_train[:max_samples]
        images = images[:int(len(images) * 0.85)]
        if return_labels:
            labels = y_train[:len(images)]
            num_classes = len(np.unique(labels))
    else:
        images = x_train[:max_samples]
        images = images[int(len(images) * 0.85):]
        if return_labels:
            labels = y_train[:max_samples]
            labels = labels[int(len(labels) * 0.85):]
            num_classes = len(np.unique(labels))

    # Check the data shape
    print(f"{split} data: {len(images)}")
    if return_labels:
        print(f"Number of classes: {num_classes}")

    return images, labels, num_classes

def SVHNLoader(split='train', max_samples=None, return_labels=False):
    from torchvision import datasets, transforms

    labels = None
    num_classes = None

    # Define a transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training set
    train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    x_train = train_dataset.data
    y_train = train_dataset.labels

    # Download and load the test set
    test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    x_test = test_dataset.data
    y_test = test_dataset.labels

    if split == 'test':
        images = x_test[:max_samples]
        if return_labels:
            labels = y_test[:max_samples]
            num_classes = len(np.unique(labels))
    elif split == 'train':
        images = x_train[:max_samples]
        images = images[:int(len(images) * 0.85)]
        if return_labels:
            labels = y_train[:len(images)]
            num_classes = len(np.unique(labels))
    else:
        images = x_train[:max_samples]
        images = images[int(len(images) * 0.85):]
        if return_labels:
            labels = y_train[:max_samples]
            labels = labels[int(len(labels) * 0.85):]
            num_classes = len(np.unique(labels))

    # Check the data shape
    print(f"{split} data: {len(images)}")
    if return_labels:
        print(f"Number of classes: {num_classes}")

    return images, labels, num_classes

def TestLoader(root_dir='trian', split='val'):
    """some dataset loader with lazy loading(only load file paths)"""

    split_dir = Path(root_dir)
    image_paths = []

    if not split_dir.exists():
        raise ValueError(f"directory not found: {split_dir}")

    image_files = list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpg"))
    image_paths.extend(image_files)

    print(f"{split} dataset: {len(image_paths)} samples")

    return image_paths, None, None
