# ALU Machine Learning

A comprehensive repository for machine learning implementations covering supervised learning with focus on binary classification and optimization techniques.

## Repository Structure

```
alu-machine_learning/
├── supervised_learning/
│   ├── classification/
│   │   ├── 0-neuron.py          # Basic Neuron class with properties
│   │   ├── 1-neuron.py          # Neuron with read-only properties
│   │   ├── 2-neuron.py          # Neuron with forward propagation
│   │   ├── 3-neuron.py          # Neuron with cost calculation
│   │   └── 4-neuron.py          # Neuron with evaluation method
│   └── optimization/
│       └── 0-norm_constants.py  # Normalization constants function
└── README.md
```

## Classification Module

The classification module implements a single neuron for binary classification with progressive feature additions.

### 0-neuron.py - Basic Neuron Class

Defines a basic neuron with private attributes and properties.

**Features:**
- Private attributes: `__W` (weights), `__b` (bias), `__A` (activation)
- Property getters for W, b, A
- Setter for A attribute
- Input validation

**Usage:**
```python
import numpy as np
from supervised_learning.classification.neuron import Neuron

np.random.seed(0)
neuron = Neuron(nx=784)  # 784 input features
print(neuron.W.shape)    # (1, 784)
print(neuron.b)         # 0
print(neuron.A)         # 0
```

### 1-neuron.py - Read-Only Neuron

Extends the basic neuron with read-only properties (no setter for A).

**Features:**
- All properties are read-only
- Attempting to modify A raises AttributeError
- Strict input validation

**Usage:**
```python
neuron = Neuron(nx=784)
neuron.A = 10  # Raises: AttributeError: can't set attribute
```

### 2-neuron.py - Forward Propagation

Adds forward propagation method using sigmoid activation.

**Features:**
- `forward_prop(X)` method
- Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
- Updates private __A attribute
- Returns activated output

**Forward Propagation Formula:**
```
z = W · X + b
A = 1 / (1 + e^(-z))
```

**Usage:**
```python
neuron = Neuron(nx=784)
A = neuron.forward_prop(X)  # X shape: (784, m), returns (1, m)
```

### 3-neuron.py - Cost Calculation

Adds cost calculation using binary cross-entropy loss.

**Features:**
- `cost(Y, A)` method
- Binary cross-entropy loss formula
- Protection against log(0) using 1.0000001

**Cost Formula:**
```
Cost = -1/m * Σ(Y * log(A) + (1 - Y) * log(1.0000001 - A))
```

**Usage:**
```python
neuron = Neuron(nx=784)
A = neuron.forward_prop(X)
cost = neuron.cost(Y, A)
print(cost)  # 4.365104944262272
```

### 4-neuron.py - Evaluation

Adds evaluation method for predictions and cost.

**Features:**
- `evaluate(X, Y)` method
- Binary predictions (0 or 1)
- Returns prediction and cost

**Prediction Logic:**
- If A >= 0.5 → predict 1
- If A < 0.5 → predict 0

**Usage:**
```python
neuron = Neuron(nx=784)
prediction, cost = neuron.evaluate(X, Y)
print(prediction)  # [[0 0 0 ... 0 0 0]]
print(cost)        # 4.365104944262272
```

## Optimization Module

The optimization module provides utility functions for data preprocessing.

### 0-norm_constants.py - Normalization Constants

Calculates standardization constants for data normalization.

**Features:**
- `normalization_constants(X)` function
- Calculates mean and standard deviation per feature
- Essential for data preprocessing

**Usage:**
```python
import numpy as np
from supervised_learning.optimization.norm_constants import normalization_constants

np.random.seed(0)
a = np.random.normal(0, 2, size=(100, 1))
b = np.random.normal(2, 1, size=(100, 1))
c = np.random.normal(-3, 10, size=(100, 1))
X = np.concatenate((a, b, c), axis=1)

mean, std = normalization_constants(X)
print(mean)  # [ 0.11961603  2.08201297 -3.59232261]
print(std)   # [2.01576449 1.034667   9.52002619]
```

## Class Hierarchy

```
Neuron (0-neuron.py)
   ↓
Neuron (1-neuron.py)  - Read-only properties
   ↓
Neuron (2-neuron.py)  + forward_prop()
   ↓
Neuron (3-neuron.py)  + cost()
   ↓
Neuron (4-neuron.py)  + evaluate()
```

## Requirements

- Python 3.x
- NumPy >= 1.14.0

## Installation

```bash
# Clone the repository
git clone https://github.com/ikalisa1/alu-machine_learning.git

# Navigate to the directory
cd alu-machine_learning

# Install dependencies
pip install numpy
```

## Example: Complete Workflow

```python
#!/usr/bin/env python3

import numpy as np
from supervised_learning.classification.neuron import Neuron

# Load or create data
np.random.seed(0)
X = np.random.randn(784, 100)  # 100 samples, 784 features
Y = np.random.randint(0, 2, (1, 100))  # Binary labels

# Create and train neuron
neuron = Neuron(nx=784)

# Forward propagation
A = neuron.forward_prop(X)

# Calculate cost
cost = neuron.cost(Y, A)
print(f"Cost: {cost}")

# Evaluate predictions
predictions, eval_cost = neuron.evaluate(X, Y)
print(f"Predictions shape: {predictions.shape}")
print(f"Evaluation cost: {eval_cost}")
```

## Key Concepts

### Binary Classification
Uses sigmoid activation function to produce probabilities between 0 and 1.

### Sigmoid Activation
- Formula: σ(z) = 1 / (1 + e^(-z))
- Output range: (0, 1)
- Used for probability estimation

### Binary Cross-Entropy Loss
- Measures difference between predicted and actual labels
- Lower cost = better predictions
- Formula: -[Y*log(A) + (1-Y)*log(1-A)]

### Standardization
- Normalization: scales features to have mean=0, std=1
- Improves model training stability
- Applied before training

## Author

Created for ALU (African Leadership University) Machine Learning Program

## License

MIT License - Feel free to use and modify

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Changelog

### Version 1.0
- Added 0-neuron.py: Basic neuron class
- Added 1-neuron.py: Read-only properties
- Added 2-neuron.py: Forward propagation
- Added 3-neuron.py: Cost calculation
- Added 4-neuron.py: Evaluation method
- Added 0-norm_constants.py: Normalization utilities
