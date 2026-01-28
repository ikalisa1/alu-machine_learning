# Optimization Module

This module contains optimization and data preprocessing utilities for machine learning models.

## Contents

### 0-norm_constants.py

**Function:** `normalization_constants(X)`

Calculates the normalization (standardization) constants of a matrix.

#### Parameters

- **X** (numpy.ndarray): Matrix of shape (m, nx) to normalize
  - m: Number of data points (rows)
  - nx: Number of features (columns)

#### Returns

- **tuple**: (mean, std)
  - **mean** (numpy.ndarray): Mean of each feature, shape (nx,)
  - **std** (numpy.ndarray): Standard deviation of each feature, shape (nx,)

#### Description

This function calculates the standardization constants (mean and standard deviation) for each feature in the dataset. These constants are essential for normalizing data before training machine learning models.

Normalization helps:
- Scale features to similar ranges
- Improve model training stability
- Accelerate convergence during training
- Prevent features with large scales from dominating the model

#### Formula

```
mean = (1/m) * Σ(x_i) for each feature
std = sqrt((1/m) * Σ(x_i - mean)²) for each feature
```

#### Usage Example

```python
import numpy as np
from normalization_constants import normalization_constants

# Create sample data
np.random.seed(0)
a = np.random.normal(0, 2, size=(100, 1))      # Mean≈0, Std≈2
b = np.random.normal(2, 1, size=(100, 1))      # Mean≈2, Std≈1
c = np.random.normal(-3, 10, size=(100, 1))    # Mean≈-3, Std≈10
X = np.concatenate((a, b, c), axis=1)          # Shape: (100, 3)

# Get normalization constants
mean, std = normalization_constants(X)

print("Mean:", mean)
# Output: [ 0.11961603  2.08201297 -3.59232261]

print("Std:", std)
# Output: [2.01576449 1.034667   9.52002619]
```

#### Using for Normalization

After obtaining the normalization constants, you can normalize your data:

```python
# Normalize the training data
X_normalized = (X - mean) / std

# Normalize test data (using training constants!)
X_test_normalized = (X_test - mean) / std
```

**Important:** Always use the constants calculated from the training set to normalize both training and test data.

#### Example with Different Data Distributions

```python
import numpy as np
from normalization_constants import normalization_constants
import matplotlib.pyplot as plt

# Feature 1: Small scale (0-10)
feature1 = np.random.uniform(0, 10, 1000)

# Feature 2: Large scale (0-10000)
feature2 = np.random.uniform(0, 10000, 1000)

# Feature 3: Negative scale (-100 to 100)
feature3 = np.random.uniform(-100, 100, 1000)

X = np.column_stack((feature1, feature2, feature3))

mean, std = normalization_constants(X)

print(f"Feature 1 - Mean: {mean[0]:.2f}, Std: {std[0]:.2f}")
print(f"Feature 2 - Mean: {mean[1]:.2f}, Std: {std[1]:.2f}")
print(f"Feature 3 - Mean: {mean[2]:.2f}, Std: {std[2]:.2f}")

# Normalize
X_normalized = (X - mean) / std

# Now all features have mean≈0 and std≈1
print(f"\nNormalized means: {np.mean(X_normalized, axis=0)}")
print(f"Normalized stds: {np.std(X_normalized, axis=0)}")
```

## Notes

- Uses NumPy's `mean()` and `std()` functions with `axis=0`
- Computes statistics across all data points (rows) for each feature (column)
- Returns arrays suitable for element-wise operations
- Works with any number of features
- Handles both 1D and 2D arrays

## Related Functions

For a complete machine learning pipeline, this function is typically used alongside:
- Model training functions (from classification module)
- Data loading utilities
- Cross-validation techniques
- Evaluation metrics

## Requirements

- NumPy >= 1.14.0

## Author

Created for ALU Machine Learning Program

## License

MIT License
