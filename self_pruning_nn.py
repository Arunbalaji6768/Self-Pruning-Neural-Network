"""
Self-Pruning Neural Network on CIFAR-10
========================================
A neural network that learns to prune its own weights during training
using learnable gate parameters with L1 sparsity regularization.

Key Concepts:
    - PrunableLinear: Custom linear layer with learnable gate scores
    - Sparsity Loss: L1 penalty on gate values to encourage pruning
    - Automatic Pruning: Gates converge to 0 -> weights are effectively removed
"""

import os
import sys
import time
import json

# Force line-buffered stdout on Windows so training progress is visible
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tabulate import tabulate


# ============================================================================
# Part 1: Custom PrunableLinear Layer
# ============================================================================

class PrunableLinear(nn.Module):
    """
    A linear layer with learnable gate scores for automatic pruning.

    Each weight w_ij has a corresponding gate score g_ij. During the forward
    pass, the effective weight is: w_ij * sigmoid(g_ij). When the gate score
    is driven toward large negative values by the sparsity loss, sigmoid(g_ij)
    approaches 0, effectively pruning that connection.

    Parameters:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        bias (bool): If True, adds a learnable bias. Default: True

    Attributes:
        weight (nn.Parameter): Learnable weight matrix of shape (out_features, in_features)
        gate_scores (nn.Parameter): Learnable gate scores of shape (out_features, in_features)
        bias (nn.Parameter): Learnable bias of shape (out_features), if bias=True

    Example:
        >>> layer = PrunableLinear(784, 256)
        >>> x = torch.randn(32, 784)
        >>> output = layer(x)
        >>> print(output.shape)
        torch.Size([32, 256])
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PrunableLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Standard learnable weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Learnable gate scores - initialized to 0.0 so sigmoid(0.0) = 0.5
        # This neutral starting point lets gates be pushed toward 0 (pruned)
        # or toward 1 (kept) based on the balance of CE loss and sparsity penalty.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming uniform initialization for the weight matrix."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weights.

        Computation:
            gates = sigmoid(gate_scores)        # in (0, 1)
            pruned_weights = weight * gates      # Element-wise masking
            output = x @ pruned_weights.T + bias # Standard linear

        Both weight and gate_scores receive gradients via autograd.
        """
        gates = torch.sigmoid(self.gate_scores)         # Differentiable gating
        pruned_weights = self.weight * gates             # Apply soft mask
        return F.linear(x, pruned_weights, self.bias)    # Linear transformation

    def get_gate_values(self) -> torch.Tensor:
        """Return current gate values (sigmoid of gate_scores), detached from graph."""
        return torch.sigmoid(self.gate_scores).detach()

    def get_sparsity(self, threshold: float = 0.01) -> float:
        """
        Calculate the percentage of effectively pruned connections.

        Args:
            threshold: Gate value below which a connection is considered pruned.

        Returns:
            Sparsity percentage (0-100).
        """
        gate_values = self.get_gate_values()
        total = gate_values.numel()
        pruned = (gate_values < threshold).sum().item()
        return (pruned / total) * 100.0

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ============================================================================
# Part 2: Self-Pruning Neural Network Architecture
# ============================================================================

class SelfPruningNetwork(nn.Module):
    """
    A convolutional neural network for CIFAR-10 with prunable fully-connected layers.

    Architecture:
        Conv Block 1: Conv2d(3,32) -> BN -> ReLU -> Conv2d(32,32) -> BN -> ReLU -> MaxPool -> Dropout
        Conv Block 2: Conv2d(32,64) -> BN -> ReLU -> Conv2d(64,64) -> BN -> ReLU -> MaxPool -> Dropout
        FC Block:     PrunableLinear(64*8*8, 512) -> ReLU -> Dropout -> PrunableLinear(512, 256) -> ReLU -> Dropout -> PrunableLinear(256, 10)

    The convolutional layers extract spatial features, while the prunable
    fully-connected layers learn which connections are important.
    """

    def __init__(self):
        super(SelfPruningNetwork, self).__init__()

        # --- Convolutional Feature Extractor ---
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )

        # --- Prunable Fully-Connected Classifier ---
        self.fc1 = PrunableLinear(64 * 8 * 8, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

        self.dropout_fc = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        # Flatten for FC layers
        x = x.view(x.size(0), -1)

        # Prunable classifier
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)

        return x

    def get_prunable_layers(self) -> list:
        """Return all PrunableLinear layers in the network."""
        return [module for module in self.modules() if isinstance(module, PrunableLinear)]

    def get_all_gate_values(self) -> torch.Tensor:
        """Concatenate all gate values across all prunable layers."""
        gate_values = []
        for layer in self.get_prunable_layers():
            gate_values.append(layer.get_gate_values().cpu().flatten())
        return torch.cat(gate_values)

    def get_overall_sparsity(self, threshold: float = 0.01) -> float:
        """Calculate overall sparsity across all prunable layers."""
        all_gates = self.get_all_gate_values()
        total = all_gates.numel()
        pruned = (all_gates < threshold).sum().item()
        return (pruned / total) * 100.0

    def get_layer_sparsities(self, threshold: float = 0.01) -> dict:
        """Return per-layer sparsity percentages."""
        sparsities = {}
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                sparsities[name] = module.get_sparsity(threshold)
        return sparsities

    def count_parameters(self) -> dict:
        """Count total and prunable parameters."""
        total = sum(p.numel() for p in self.parameters())
        prunable_weights = sum(
            m.weight.numel() for m in self.get_prunable_layers()
        )
        prunable_gates = sum(
            m.gate_scores.numel() for m in self.get_prunable_layers()
        )
        return {
            "total": total,
            "prunable_weights": prunable_weights,
            "prunable_gate_params": prunable_gates,
            "non_prunable": total - prunable_weights - prunable_gates,
        }


# ============================================================================
# Part 3: Sparsity-Regularized Loss Function
# ============================================================================

class SparsityLoss(nn.Module):
    """
    Combined loss: CrossEntropy + lambda * L1(gate_values)

    The L1 penalty on sigmoid(gate_scores) pushes gate values toward 0,
    effectively pruning unimportant connections. Higher lambda -> more pruning.

    Args:
        lambda_sparse (float): Weight of the sparsity penalty.
    """

    def __init__(self, lambda_sparse: float):
        super(SparsityLoss, self).__init__()
        self.lambda_sparse = lambda_sparse
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        model: SelfPruningNetwork,
    ) -> tuple:
        """
        Compute total loss with sparsity regularization.

        Returns:
            (total_loss, ce_loss_value, sparsity_loss_value)
        """
        ce_loss = self.cross_entropy(outputs, targets)

        # L1 penalty: sum of all gate values across all prunable layers
        sparsity_loss = torch.tensor(0.0, device=outputs.device)
        for layer in model.get_prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)   # Recompute in graph
            sparsity_loss = sparsity_loss + gates.sum()

        total_loss = ce_loss + self.lambda_sparse * sparsity_loss

        return total_loss, ce_loss.item(), sparsity_loss.item()


# ============================================================================
# Data Loading
# ============================================================================

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 0) -> tuple:
    """
    Prepare CIFAR-10 train and test data loaders with standard augmentation.

    Training augmentation:
        - Random horizontal flip
        - Random crop with padding
        - Normalization to mean=0, std=1 per channel

    Returns:
        (train_loader, test_loader, classes)
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda,
    )

    classes = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )

    return train_loader, test_loader, classes


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_one_epoch(
    model: SelfPruningNetwork,
    train_loader: DataLoader,
    criterion: SparsityLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch. Returns metrics dict."""
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_sparse = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss, ce_val, sparse_val = criterion(outputs, targets, model)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ce += ce_val
        running_sparse += sparse_val

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    n = len(train_loader)
    return {
        "loss": running_loss / n,
        "ce_loss": running_ce / n,
        "sparse_loss": running_sparse / n,
        "accuracy": 100.0 * correct / total,
    }


def evaluate(
    model: SelfPruningNetwork,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on test set. Returns metrics dict."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    sparsity = model.get_overall_sparsity(threshold=0.01)
    layer_sparsities = model.get_layer_sparsities(threshold=0.01)

    return {
        "accuracy": accuracy,
        "sparsity": sparsity,
        "layer_sparsities": layer_sparsities,
    }


def train_model(
    lambda_sparse: float,
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 128,
    device: torch.device = None,
) -> dict:
    """
    Full training pipeline for a given sparsity coefficient lambda.

    Args:
        lambda_sparse: Sparsity penalty weight
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Compute device

    Returns:
        Dictionary with model, history, and final metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print(f"  TRAINING WITH lambda = {lambda_sparse}")
    print(f"  Device: {device} | Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")
    print("=" * 70 + "\n")

    # Data
    train_loader, test_loader, classes = get_cifar10_loaders(batch_size)

    # Model
    model = SelfPruningNetwork().to(device)
    param_info = model.count_parameters()
    print(f"  Total parameters:      {param_info['total']:,}")
    print(f"  Prunable weights:      {param_info['prunable_weights']:,}")
    print(f"  Prunable gate params:  {param_info['prunable_gate_params']:,}")
    print(f"  Non-prunable params:   {param_info['non_prunable']:,}")
    print()

    # Loss & Optimizer
    # Use separate param groups: gate_scores get a higher LR so the
    # L1 sparsity penalty can actually push them toward zero quickly.
    criterion = SparsityLoss(lambda_sparse)
    gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "gate_scores" in name:
            gate_params.append(param)
        else:
            other_params.append(param)
    optimizer = optim.Adam([
        {"params": other_params, "lr": lr, "weight_decay": 1e-4},
        {"params": gate_params, "lr": lr * 10, "weight_decay": 0},  # 10x LR for gates
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    history = {
        "train_loss": [], "train_acc": [], "test_acc": [],
        "sparsity": [], "ce_loss": [], "sparse_loss": [],
    }

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Evaluate
        eval_metrics = evaluate(model, test_loader, device)

        # Update scheduler
        scheduler.step()

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["test_acc"].append(eval_metrics["accuracy"])
        history["sparsity"].append(eval_metrics["sparsity"])
        history["ce_loss"].append(train_metrics["ce_loss"])
        history["sparse_loss"].append(train_metrics["sparse_loss"])

        # Track best
        if eval_metrics["accuracy"] > best_acc:
            best_acc = eval_metrics["accuracy"]

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"CE: {train_metrics['ce_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Test Acc: {eval_metrics['accuracy']:.2f}% | "
                f"Sparsity: {eval_metrics['sparsity']:.2f}% | "
                f"Time: {elapsed:.0f}s"
            )

    total_time = time.time() - start_time

    # Final evaluation
    final_metrics = evaluate(model, test_loader, device)

    print(f"\n  --- Final Results for lambda = {lambda_sparse} ---")
    print(f"  Best Test Accuracy:   {best_acc:.2f}%")
    print(f"  Final Test Accuracy:  {final_metrics['accuracy']:.2f}%")
    print(f"  Overall Sparsity:     {final_metrics['sparsity']:.2f}%")
    print(f"  Training Time:        {total_time:.1f}s")
    print(f"  Layer Sparsities:")
    for layer_name, sp in final_metrics["layer_sparsities"].items():
        print(f"    {layer_name}: {sp:.2f}%")

    return {
        "model": model,
        "history": history,
        "final_accuracy": final_metrics["accuracy"],
        "best_accuracy": best_acc,
        "final_sparsity": final_metrics["sparsity"],
        "layer_sparsities": final_metrics["layer_sparsities"],
        "training_time": total_time,
        "lambda": lambda_sparse,
        "gate_values": model.get_all_gate_values().cpu().numpy(),
    }


# ============================================================================
# Part 4: Visualization & Reporting
# ============================================================================

def plot_gate_histograms(results: list, save_path: str = "gate_histograms.png"):
    """
    Plot histograms of gate values for each lambda configuration.
    High sparsity -> spike near 0; low lambda -> more spread distribution.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))

    if len(results) == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF9800", "#F44336"]

    for idx, (result, color) in enumerate(zip(results, colors)):
        ax = axes[idx]
        gate_vals = result["gate_values"]
        lam = result["lambda"]

        ax.hist(
            gate_vals, bins=100, color=color, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5, label="Pruning threshold (0.01)")
        ax.set_title(
            f"lambda = {lam}\n"
            f"Accuracy: {result['final_accuracy']:.2f}%  |  "
            f"Sparsity: {result['final_sparsity']:.2f}%",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Gate Value (sigmoid of gate_score)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Distribution of Gate Values Across Different Sparsity Penalties (lambda)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [OK] Gate histogram saved to: {save_path}")


def plot_training_curves(results: list, save_path: str = "training_curves.png"):
    """Plot training curves (accuracy, loss, sparsity) for all lambda values."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#2196F3", "#FF9800", "#F44336"]

    for result, color in zip(results, colors):
        lam = result["lambda"]
        history = result["history"]
        epochs = range(1, len(history["train_acc"]) + 1)

        # Test Accuracy
        axes[0, 0].plot(epochs, history["test_acc"], color=color,
                        label=f"lambda={lam}", linewidth=2)
        axes[0, 0].set_title("Test Accuracy vs Epoch", fontweight="bold")
        axes[0, 0].set_ylabel("Accuracy (%)")

        # Train Loss
        axes[0, 1].plot(epochs, history["train_loss"], color=color,
                        label=f"lambda={lam}", linewidth=2)
        axes[0, 1].set_title("Total Training Loss vs Epoch", fontweight="bold")
        axes[0, 1].set_ylabel("Loss")

        # Sparsity
        axes[1, 0].plot(epochs, history["sparsity"], color=color,
                        label=f"lambda={lam}", linewidth=2)
        axes[1, 0].set_title("Sparsity (% gates < 0.01) vs Epoch", fontweight="bold")
        axes[1, 0].set_ylabel("Sparsity (%)")

        # CE Loss
        axes[1, 1].plot(epochs, history["ce_loss"], color=color,
                        label=f"lambda={lam}", linewidth=2)
        axes[1, 1].set_title("Cross-Entropy Loss vs Epoch", fontweight="bold")
        axes[1, 1].set_ylabel("CE Loss")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Training Dynamics: Self-Pruning Neural Network on CIFAR-10",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Training curves saved to: {save_path}")


def print_results_table(results: list):
    """Print a formatted comparison table of all experiments."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 70 + "\n")

    headers = [
        "Lambda (Sparsity Penalty)",
        "Test Accuracy (%)",
        "Sparsity (%)",
        "Training Time (s)",
    ]
    rows = []
    for r in results:
        rows.append([
            f"{r['lambda']:.1e}",
            f"{r['final_accuracy']:.2f}",
            f"{r['final_sparsity']:.2f}",
            f"{r['training_time']:.1f}",
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))

    # Per-layer breakdown
    print(f"\n  Per-Layer Sparsity Breakdown:")
    layer_headers = ["Layer"] + [f"lambda={r['lambda']:.1e}" for r in results]
    layer_names = list(results[0]["layer_sparsities"].keys())
    layer_rows = []
    for name in layer_names:
        row = [name]
        for r in results:
            row.append(f"{r['layer_sparsities'][name]:.2f}%")
        layer_rows.append(row)

    print(tabulate(layer_rows, headers=layer_headers, tablefmt="grid", stralign="center"))


def save_results_json(results: list, save_path: str = "experiment_results.json"):
    """Save experiment results to JSON for reproducibility."""
    json_data = []
    for r in results:
        entry = {
            "lambda": r["lambda"],
            "final_accuracy": r["final_accuracy"],
            "best_accuracy": r["best_accuracy"],
            "final_sparsity": r["final_sparsity"],
            "layer_sparsities": r["layer_sparsities"],
            "training_time": r["training_time"],
            "history": r["history"],
        }
        json_data.append(entry)

    with open(save_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  [OK] Results JSON saved to: {save_path}")


# ============================================================================
# Main - Run All Experiments
# ============================================================================

def main():
    """
    Run the full self-pruning neural network experiment on CIFAR-10.

    Three lambda values are tested:
        - lambda = 1e-5 (Low)    -> Minimal pruning, highest accuracy
        - lambda = 1e-4 (Medium) -> Moderate pruning, balanced accuracy
        - lambda = 1e-3 (High)   -> Aggressive pruning, lower accuracy
    """
    print("\n" + "=" * 70)
    print("  SELF-PRUNING NEURAL NETWORK - CIFAR-10 EXPERIMENT")
    print("=" * 70)

    # Configuration
    LAMBDA_VALUES = [1e-4, 1e-3, 1e-2]   # Low / Medium / High sparsity
    EPOCHS = 15
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  Lambda values: {LAMBDA_VALUES}")
    print(f"  Epochs: {EPOCHS}")

    # Run experiments
    all_results = []
    for lam in LAMBDA_VALUES:
        result = train_model(
            lambda_sparse=lam,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            device=device,
        )
        all_results.append(result)

    # Results & Visualization
    print_results_table(all_results)
    plot_gate_histograms(all_results, save_path="gate_histograms.png")
    plot_training_curves(all_results, save_path="training_curves.png")
    save_results_json(all_results, save_path="experiment_results.json")

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("  Outputs: gate_histograms.png, training_curves.png, experiment_results.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
