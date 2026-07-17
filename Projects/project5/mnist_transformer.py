# Shriman Raghav Srinivasan
# CS 5330: Pattern Recognition & Computer Vision
# Project 5: Recognition using Deep Networks - Task 4

import sys
import os
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class NetConfig:
    def __init__(self,
                 name = 'vit_base',
                 dataset = 'mnist',
                 patch_size = 4,
                 stride = 2,
                 embed_dim = 48,
                 depth = 4,
                 num_heads = 8,
                 mlp_dim = 128,
                 dropout = 0.1,
                 use_cls_token = False,  
                 epochs = 15,
                 batch_size = 64,
                 lr = 1e-3,
                 weight_decay = 1e-4,
                 seed = 0,
                 optimizer = 'adamw',
                 device = 'cuda',  # Will override to cuda if available
                 ):

        # data set fixed attributes
        self.image_size = 28
        self.in_channels = 1
        self.num_classes = 10

        # variable things
        self.name = name
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_cls_token = use_cls_token
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.optimizer = optimizer
        
        # Override device selection based on system availability
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        s = "Name,Dataset,PatchSize,Stride,Dim,Depth,Heads,MLPDim,Dropout,CLS,Epochs,Batch,LR,Decay,Seed,Optimizer,TestAcc,BestEpoch\n"
        s += "%s,%s,%d,%d,%d,%d,%d,%d,%0.2f,%s,%d,%d,%f,%f,%d,%s," % (
            self.name,
            self.dataset,
            self.patch_size,
            self.stride,
            self.embed_dim,
            self.depth,
            self.num_heads,
            self.mlp_dim,
            self.dropout,
            self.use_cls_token,
            self.epochs,
            self.batch_size,
            self.lr,
            self.weight_decay,
            self.seed,
            self.optimizer
            )
        self.config_string = s

        return

# Patch Embedding class
class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.
    """
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            stride: int,
            in_channels: int,
            embed_dim: int,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=stride,
        )

        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)
        self.num_patches = self._compute_num_patches()

    def _compute_num_patches(self) -> int:
        positions_per_dim = ((self.image_size - self.patch_size) // self.stride) + 1
        return positions_per_dim * positions_per_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: extract patches, shape = (B, patch_dim, N)
        x = self.unfold(x)
        # Step 2: transpose, shape = (B, N, patch_dim)
        x = x.transpose(1, 2)
        # Step 3: project, shape = (B, N, embed_dim)
        x = self.proj(x)
        return x

# The Transformer Network class
class NetTransformer(nn.Module):
    def __init__(self, config):
        super(NetTransformer, self).__init__()

        # make the patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        # how many tokens are there?
        num_tokens = self.patch_embed.num_patches
        print("Number of tokens: %d" % (num_tokens) )

        # does it use a classifier token or a global average token?
        self.use_cls_token = config.use_cls_token

        # if it uses a classifier node, create a source for the node
        if self.use_cls_token:
            self.cls_token = nn.Parameter( torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens+1
        else: # no CLS token
            self.cls_token = None
            total_tokens = num_tokens

        # need to include a learned positional embedding, one for each token
        self.pos_embed = nn.Parameter(
            torch.zeros( 1, total_tokens, config.embed_dim ) )
        self.pos_dropout = nn.Dropout( config.dropout )

        # Use the Torch Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = config.embed_dim,
            nhead = config.num_heads,
            dim_feedforward = config.mlp_dim,
            dropout = config.dropout,
            activation = 'gelu',
            batch_first = True,
            norm_first = True,
        )

        # Create a stack of transformer layers to build an encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers = config.depth,
        )

        # final normalization layer prior to classification
        self.norm = nn.LayerNorm( config.embed_dim )

        # linear layer for classification
        self.classifier = nn.Sequential(
            nn.Linear( config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear( config.mlp_dim, config.num_classes )
        )

        # Initialize parameter weights
        self._init_parameters()
        return

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std = .02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std = 0.02 )

    def forward( self, x ):
        # 1. Execute the patch embedding layer
        x = self.patch_embed(x)

        # 2. Get the batch size
        batch_size = x.size(0)

        # 3. Add the optional CLS token to the set 
        if self.use_cls_token:
            cls_token = self.cls_token.expand( batch_size, -1, -1 )
            x = torch.cat( [cls_token, x], dim = 1 )

        # 4. Add the learnable positional embedding to each token
        x = x + self.pos_embed

        # 5. Run the dropout layer right after positional embedding
        x = self.pos_dropout(x)
        
        # 6. Run the transformer encoder
        x = self.encoder(x)

        # 7. Either pool the tokens or use the cls token (first token)
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        # 8. Final normalization of the token to classify
        x = self.norm(x)

        # 9. Call the classification MLP
        x = self.classifier(x)

        # 10. Return the log-softmax of the output layer
        return F.log_softmax( x, dim=1 )

# Training utility function
def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total_loss = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += len(data)
        
    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    error_rate = 100. - accuracy
    return avg_loss, error_rate

# Evaluation utility function
def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)
            
    avg_loss = test_loss / total_samples
    accuracy = 100. * correct / total_samples
    error_rate = 100. - accuracy
    return avg_loss, error_rate

# Main run function
def main(argv):
    # Setup configuration
    config = NetConfig()
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading MNIST dataset...")
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=config.batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Instantiate Model
    model = NetTransformer(config).to(device)
    
    # Setup optimizer
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
        
    train_errors = []
    test_errors = []
    train_losses = []
    test_losses = []
    epoch_times = []
    
    print("\nStarting Transformer training on MNIST...")
    print("-" * 50)
    
    total_start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_err = train_epoch(model, device, train_loader, optimizer, epoch)
        test_loss, test_err = test_model(model, device, test_loader)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_errors.append(train_err)
        test_errors.append(test_err)
        
        print(f"Epoch {epoch:2d}/{config.epochs:2d} | "
              f"Train Loss: {train_loss:.4f} (Err: {train_err:.2f}%) | "
              f"Test Loss: {test_loss:.4f} (Err: {test_err:.2f}%) | "
              f"Time: {epoch_time:.2f}s")
              
    total_time = time.time() - total_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    print("-" * 50)
    print(f"Training completed in {total_time:.2f}s")
    print(f"Average epoch training time: {avg_epoch_time:.2f}s")
    print(f"Final Test Error Rate: {test_errors[-1]:.2f}% (Accuracy: {100.0 - test_errors[-1]:.2f}%)")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/transformer_model.pth")
    print("Saved models/transformer_model.pth")
    
    # Plotting Curves
    epochs_range = range(1, config.epochs + 1)
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(epochs_range, test_losses, label='Test Loss', color='orange', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MNIST Transformer Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("transformer_loss_plot.png")
    plt.close()
    
    # Error rate plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_errors, label='Train Error Rate (%)', color='blue', marker='o')
    plt.plot(epochs_range, test_errors, label='Test Error Rate (%)', color='orange', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate (%)')
    plt.title('MNIST Transformer Error Rate Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("transformer_error_plot.png")
    plt.close()
    print("Saved transformer_loss_plot.png and transformer_error_plot.png")
    
    # Print comparison with CNN
    print("\n" + "=" * 50)
    print("Performance Comparison: CNN vs. Transformer")
    print("=" * 50)
    # CNN metrics from task 1 walkthrough: 5 epochs, ~1.46s per epoch on CUDA, final test accuracy ~98.18%
    print("CNN Model (Task 1):")
    print("  - Total Epochs: 5")
    print("  - Average Epoch Time: ~1.46s (on GPU)")
    print("  - Final Test Accuracy: 98.18%")
    print("  - Final Test Error: 1.82%")
    print("\nTransformer Model (Task 4):")
    print(f"  - Total Epochs: 15")
    print(f"  - Average Epoch Time: {avg_epoch_time:.2f}s (on {device})")
    print(f"  - Final Test Accuracy: {100.0 - test_errors[-1]:.2f}%")
    print(f"  - Final Test Error: {test_errors[-1]:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main(sys.argv)
