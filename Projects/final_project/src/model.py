import torch
import torch.nn as nn
import torchvision.models as models

def set_seed(seed):
    """
    Sets the random seed for reproducibility across python, numpy, and pytorch.
    """
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior if using GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_model(condition, num_classes=6, seed=None):
    """
    Loads ResNet-18 and configures the architecture, initialization, and freezing
    according to the specified training condition.
    
    Args:
        condition (str): One of 'pretrained-frozen', 'random-frozen', or 'random-full'.
        num_classes (int): Number of target output classes.
        seed (int): Random seed to initialize weights.
        
    Returns:
        model (nn.Module): Configured PyTorch model.
    """
    # Set seed before initializing randomly initialized parts of the network
    set_seed(seed)
    
    if condition == 'pretrained-frozen':
        # Load model with ImageNet pre-trained weights
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    elif condition in ['random-frozen', 'random-full']:
        # Load model with random initialization
        model = models.resnet18(weights=None)
    else:
        raise ValueError(f"Unknown training condition: {condition}")
        
    # Replace the classification head
    # ResNet-18 fc input features is 512
    in_features = model.fc.in_features
    
    # We must seed the new fully connected layer initialization as well
    # Setting seed again right before head initialization ensures same initialization for matching seeds/heads
    set_seed(seed)
    model.fc = nn.Linear(in_features, num_classes)
    
    # Configure parameter freezing (trainable vs frozen subsets)
    if condition in ['pretrained-frozen', 'random-frozen']:
        # theta_f is everything before layer4
        # theta_t is layer4 + fc
        for name, param in model.named_parameters():
            if name.startswith('layer4') or name.startswith('fc'):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif condition == 'random-full':
        # All parameters are trainable
        for param in model.parameters():
            param.requires_grad = True
            
    return model

def enforce_bn_eval_for_frozen_layers(model):
    """
    Finds all Batch Normalization layers inside theta_f (everything before layer4)
    and puts them in evaluation mode to prevent running stats updates.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            # Check if BN layer is within the frozen subset theta_f
            if not name.startswith('layer4') and not name.startswith('fc'):
                module.eval()
