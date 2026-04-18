# 🧠 Self-Pruning Neural Network on CIFAR-10

A neural network that **learns to prune its own weights** during training using learnable gate parameters with L1 sparsity regularization.

## ✨ Key Idea

Instead of pruning after training, this network prunes **during** training. Each weight has a learnable *gate* — an L1 penalty pushes unimportant gates to zero, automatically removing unnecessary connections.

```
Forward Pass:
  gates = sigmoid(gate_scores)
  pruned_weights = weight × gates
  output = input @ pruned_weights.T + bias
```

## 🏗️ Architecture

- **Feature Extractor:** 2 conv blocks with BatchNorm, ReLU, MaxPool, Dropout
- **Classifier:** 3 `PrunableLinear` layers (4096→512→256→10) with learnable gates
- **Loss:** `CrossEntropy + λ × Σ(gate_values)` — balances accuracy & sparsity

## 📊 Experiments

Three sparsity strengths tested (results on CIFAR-10, 15 epochs):

| Lambda | Pruning Pressure | Test Accuracy | Sparsity |
|--------|------------------|---------------|----------|
| 1e-4   | Low              | 59.07%        | 76.51%   |
| 1e-3   | Medium           | 49.92%        | 95.10%   |
| 1e-2   | High             | 31.79%        | 99.68%   |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full experiment
python self_pruning_nn.py
```

### Output Files
- `gate_histograms.png` — Distribution of gate values for each λ
- `training_curves.png` — Accuracy, loss, and sparsity over training
- `experiment_results.json` — Raw results data

## 📁 Project Structure

```
├── self_pruning_nn.py     # Complete implementation
├── REPORT.md              # Detailed technical report
├── README.md              # This file
├── requirements.txt       # Dependencies
└── .gitignore             # Git ignore rules
```

## 📝 Detailed Report

See [REPORT.md](REPORT.md) for:
- Full architecture explanation
- PrunableLinear layer internals
- Sparsity loss derivation
- Results analysis & visualizations

## ⚙️ Tech Stack

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy

## 📚 References

- [Learning Both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)

---

*AI Engineering Internship Case Study — Tredence Analytics*
