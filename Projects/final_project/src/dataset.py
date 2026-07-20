import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_transforms():
    """
    Returns the standard ImageNet transform:
    - Resize so the shorter side is 224px
    - Center crop to 224x224
    - Convert to tensor
    - Normalize using ImageNet mean and standard deviation
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_dataset_splits(data_dir, batch_size=32, num_workers=4, seed=42):
    """
    Loads train and test datasets, creates a stratified 10% validation split
    from the train set, and returns DataLoader instances.
    
    Args:
        data_dir (str): Path containing 'seg_train' and 'seg_test' subdirectories.
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of worker threads for loaders.
        seed (int): Random seed for stratified splitting.
        
    Returns:
        train_loader (DataLoader), val_loader (DataLoader), test_loader (DataLoader), classes (list)
    """
    train_dir = os.path.join(data_dir, 'seg_train')
    test_dir = os.path.join(data_dir, 'seg_test')
    
    transform = get_transforms()
    
    # Load raw datasets
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    classes = full_train_dataset.classes
    targets = np.array(full_train_dataset.targets)
    
    # Stratified split: carve out 10% validation indices
    train_indices = []
    val_indices = []
    
    rng = np.random.RandomState(seed)
    
    for class_idx in range(len(classes)):
        class_indices = np.where(targets == class_idx)[0]
        n_class = len(class_indices)
        n_val = int(0.10 * n_class)
        
        # Shuffle class indices reproducibly
        shuffled = class_indices.copy()
        rng.shuffle(shuffled)
        
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])
        
    # Sort indices to maintain order
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    
    # Create Subsets
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset summary:")
    print(f"  Total classes: {len(classes)} ({classes})")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, classes
