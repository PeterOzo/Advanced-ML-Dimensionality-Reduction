### Advanced-ML-Dimensionality-Reduction

### Advanced Machine Learning: Dimensionality Reduction and Classification Performance Analysis

#### Project Overview

This repository contains a comprehensive analysis of **dimensionality reduction techniques** and their impact on machine learning classifier performance, implemented for **Advanced Machine Learning Lab 6 - Module 4**. The project systematically evaluates **Principal Component Analysis (PCA)** and **Kernel PCA** using the MNIST handwritten digit dataset, comparing their effects on training time and classification accuracy across different algorithms.

### Project Objectives

### Primary Goals
1. **Evaluate dimensionality reduction impact** on classification performance
2. **Compare PCA vs Kernel PCA** effectiveness on high-dimensional data
3. **Analyze training time trade-offs** between accuracy and computational efficiency
4. **Assess algorithm-specific responses** to dimensionality reduction
5. **Visualize high-dimensional data** using kernel transformations

### Research Questions
- How does PCA affect training time and accuracy for different classifiers?
- When is Kernel PCA superior to linear PCA?
- What are the computational trade-offs of dimensionality reduction?
- How do Random Forest and SGD classifiers respond differently to feature reduction?

---

## Dataset Description

### MNIST Handwritten Digits Dataset
- **Total Samples**: 70,000 grayscale images
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Original Dimensions**: 784 features (28×28 pixels)
- **Classes**: 10 digits (0-9)
- **Data Type**: Normalized float32 values [0, 1]

### Data Preprocessing
```python
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype('float32') / 255.0  # Normalize pixel values
y = y.astype('int')              # Convert labels to integers
```

---

## Methodology

### Experimental Design

#### 1. Baseline Classification
- **Random Forest**: 100 estimators, original 784 features
- **SGDClassifier**: Default parameters, original 784 features
- **Metrics**: Training time, test accuracy

#### 2. Linear PCA Analysis
- **Variance Preserved**: 95% of original variance
- **Dimensionality Reduction**: 784 → 154 features (80% reduction)
- **Classifiers**: Same Random Forest and SGD configurations

#### 3. Kernel PCA Analysis
- **Kernel**: Radial Basis Function (RBF)
- **Gamma Parameter**: 0.03 (optimized for effective transformation)
- **Visualization**: 2D projection for data exploration
- **Limitations**: Memory constraints required subset analysis (10,000 samples)

#### 4. Performance Evaluation
- **Training Time Measurement**: Using Python's `time` module
- **Accuracy Assessment**: Test set classification accuracy
- **Comparative Analysis**: Cross-algorithm and cross-method comparison

---

## Implementation Details

### Core Libraries and Dependencies
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import accuracy_score
import time
import seaborn as sns
```

### Memory Optimization Strategy
Due to computational constraints with Kernel PCA on the full dataset:
```python
# Memory-efficient approach for Kernel PCA
n_samples = 10000  # Reduced sample size
X_train_subset = X_train[:n_samples]
y_train_subset = y_train[:n_samples]

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.03)
X_kpca = kpca.fit_transform(X_train_subset)
```

### Dimensionality Reduction Configuration
```python
# Linear PCA (95% variance retention)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
print(f"Features reduced from 784 to {X_train_pca.shape[1]}")  # 154 components

# Kernel PCA (RBF kernel)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.03)
```

---

## Results Summary

### Comprehensive Performance Comparison

| Method | Classifier | Training Time (s) | Accuracy | Feature Count | Speed Improvement |
|--------|------------|------------------|----------|---------------|-------------------|
| **Original** | Random Forest | 38.42 | **0.9704** | 784 | Baseline |
| **Original** | SGD | 16.22 | 0.9191 | 784 | Baseline |
| **PCA (95%)** | Random Forest | 121.86 | 0.9472 | 154 | -217% (slower) |
| **PCA (95%)** | SGD | 4.25 | 0.9160 | 154 | **+281% (faster)** |

### Key Performance Metrics

#### Random Forest Analysis
- **Original Dataset**: 38.42s training, 97.04% accuracy
- **PCA Dataset**: 121.86s training, 94.72% accuracy
- **Trade-off**: 217% slower training, 2.32% accuracy loss
- **Feature Reduction**: 80% fewer features (784 → 154)

#### SGDClassifier Analysis
- **Original Dataset**: 16.22s training, 91.91% accuracy
- **PCA Dataset**: 4.25s training, 91.60% accuracy
- **Trade-off**: 281% faster training, 0.31% accuracy loss
- **Efficiency Gain**: Dramatic speed improvement with minimal accuracy cost

---

## Key Findings

### 1. Algorithm-Specific Responses to Dimensionality Reduction

#### Random Forest Behavior
- **Unexpected Result**: PCA increased training time rather than decreasing it
- **Possible Explanations**:
  - Random Forest is optimized for sparse data (original MNIST has many zeros)
  - PCA creates dense feature representations
  - Algorithm overhead dominates for reduced feature sets
  - Scikit-learn implementation characteristics

#### SGDClassifier Behavior
- **Expected Result**: Significant training time reduction with PCA
- **Explanation**: Linear algorithms benefit directly from fewer features
- **Efficiency**: 73% training time reduction with minimal accuracy loss

### 2. Dimensionality Reduction Effectiveness

#### Linear PCA Performance
- **Variance Preservation**: 95% with 80% feature reduction
- **Information Retention**: Excellent compression ratio
- **Algorithm Dependence**: Effectiveness varies by classifier type

#### Kernel PCA Insights
- **Memory Requirements**: Prohibitive for full dataset (60,000 samples)
- **Visualization Capability**: Effective for 2D data exploration
- **Practical Limitations**: Computational constraints limit applicability

### 3. Practical Implications

#### When to Use PCA
- **Linear algorithms**: Excellent speed improvements (SGD)
- **Large datasets**: Significant storage and memory savings
- **Preprocessing step**: Reduces computational complexity for subsequent analysis

#### When PCA May Not Help
- **Tree-based algorithms**: May not benefit from linear dimensionality reduction
- **Sparse data**: Original representation may be more efficient
- **Small datasets**: Overhead may outweigh benefits

---

## Technical Analysis

### Mathematical Foundation

#### PCA Transformation
```
X_reduced = X · V_k
```
Where `V_k` contains the first k eigenvectors of the covariance matrix, preserving 95% of variance.

#### Kernel PCA Transformation
```
φ(X) = K(X, X_train) · α
```
Where `K` is the RBF kernel: `K(x_i, x_j) = exp(-γ||x_i - x_j||²)`

### Computational Complexity Analysis

#### Original Dataset
- **Features**: 784
- **Training Samples**: 60,000
- **Memory Usage**: ~180 MB for training data

#### PCA-Reduced Dataset
- **Features**: 154 (80% reduction)
- **Training Samples**: 60,000
- **Memory Usage**: ~36 MB for training data (80% reduction)

### Kernel PCA Visualization Analysis
The 2D Kernel PCA projection reveals:
- **Cluster Separation**: Different digits form distinct clusters
- **Overlap Regions**: Challenging digit pairs (e.g., 4/9, 3/8)
- **Nonlinear Structure**: Captures curved decision boundaries
- **Visualization Value**: Excellent for data exploration and understanding

---

## Installation

### Prerequisites
```bash
pip install numpy matplotlib scikit-learn seaborn time
```

### Required Libraries
```
import numpy as np                    # Numerical computations
import matplotlib.pyplot as plt       # Visualization
from sklearn.datasets import fetch_openml  # Dataset loading
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.linear_model import SGDClassifier      # Stochastic Gradient Descent
from sklearn.decomposition import PCA, KernelPCA    # Dimensionality reduction
from sklearn.metrics import accuracy_score          # Performance evaluation
import time                          # Timing measurements
import seaborn as sns               # Enhanced visualization
```

---

## Usage

### Running the Complete Analysis
```python
# Execute full dimensionality reduction analysis
python mnist_dimensionality_analysis.py

# The script will:
# 1. Load and preprocess MNIST dataset
# 2. Train baseline classifiers on original data
# 3. Apply PCA transformation (95% variance)
# 4. Train classifiers on PCA-reduced data
# 5. Perform Kernel PCA visualization (subset)
# 6. Generate comprehensive performance comparison
# 7. Create visualization plots
```

### Individual Analysis Components
```python
# Load MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype('float32') / 255.0

# Apply PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X[:60000])

# Train classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
sgd = SGDClassifier(random_state=42)

# Time training and evaluate
start_time = time.time()
rf.fit(X_pca, y[:60000])
training_time = time.time() - start_time
```

### Kernel PCA Visualization
```python
# Memory-efficient Kernel PCA
n_samples = 10000
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.03)
X_kpca = kpca.fit_transform(X[:n_samples])

# Create scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y[:n_samples], cmap='viridis')
plt.colorbar(scatter)
plt.title('MNIST Kernel PCA Visualization (2D)')
```

---

## Performance Comparison

### Training Time Analysis

#### Speed Improvements by Algorithm
```python
# SGDClassifier: 73% faster with PCA
original_sgd_time = 16.22  # seconds
pca_sgd_time = 4.25        # seconds
improvement = (original_sgd_time - pca_sgd_time) / original_sgd_time * 100
print(f"SGD Speed Improvement: {improvement:.1f}%")  # 73.8%

# Random Forest: 217% slower with PCA
original_rf_time = 38.42   # seconds
pca_rf_time = 121.86       # seconds
degradation = (pca_rf_time - original_rf_time) / original_rf_time * 100
print(f"RF Speed Degradation: {degradation:.1f}%")  # 217.1%
```

### Accuracy Trade-offs

#### Accuracy Retention Analysis
```
# SGDClassifier: Minimal accuracy loss
original_sgd_acc = 0.9191
pca_sgd_acc = 0.9160
sgd_loss = (original_sgd_acc - pca_sgd_acc) * 100
print(f"SGD Accuracy Loss: {sgd_loss:.2f} percentage points")  # 0.31

# Random Forest: Moderate accuracy loss
original_rf_acc = 0.9704
pca_rf_acc = 0.9472
rf_loss = (original_rf_acc - pca_rf_acc) * 100
print(f"RF Accuracy Loss: {rf_loss:.2f} percentage points")  # 2.32
```

### Memory and Storage Benefits
- **Feature Reduction**: 80% fewer features (784 → 154)
- **Storage Savings**: 80% reduction in data storage requirements
- **Memory Usage**: Significant reduction in RAM requirements
- **Computational Efficiency**: Faster matrix operations on reduced data

---

## Visualization and Interpretation

### Training Time Comparison
The project generates comprehensive bar charts showing:
- **Training times** across all methods and algorithms
- **Accuracy comparisons** highlighting trade-offs
- **Clear visual evidence** of algorithm-specific behavior

### Kernel PCA Scatter Plot Interpretation
- **Cluster Formation**: Clear digit clusters in 2D space
- **Separation Quality**: Well-separated clusters indicate good feature extraction
- **Overlap Analysis**: Overlapping regions show classification challenges
- **Nonlinear Relationships**: Curved cluster boundaries demonstrate kernel effectiveness

---

## Academic Insights

### Theoretical Contributions

#### Dimensionality Reduction Theory
- **Curse of Dimensionality**: Demonstrates practical impact on different algorithms
- **Information Preservation**: 95% variance retention with 80% feature reduction
- **Kernel Methods**: Nonlinear transformation capabilities for complex data

#### Algorithm Analysis
- **Linear vs Tree-Based**: Fundamental differences in dimensionality reduction response
- **Computational Complexity**: Empirical validation of theoretical expectations
- **Memory Efficiency**: Practical considerations for large-scale applications

### Practical Applications

#### Machine Learning Pipeline Design
- **Preprocessing Strategy**: Informed decision-making for dimensionality reduction
- **Algorithm Selection**: Consider dimensionality reduction compatibility
- **Performance Optimization**: Balance between speed and accuracy

#### Real-World Implications
- **Computer Vision**: MNIST insights applicable to image classification tasks
- **Big Data Processing**: Dimensionality reduction for large-scale datasets
- **Resource Optimization**: Memory and computational efficiency considerations

---

## Future Improvements

### Algorithmic Enhancements
1. **Incremental PCA**: Handle larger datasets with memory constraints
2. **Sparse PCA**: Maintain sparsity while reducing dimensions
3. **Randomized PCA**: Faster computation for large datasets
4. **Adaptive Kernel Selection**: Optimize kernel parameters automatically

### Experimental Extensions
1. **Cross-Validation**: Robust performance estimation
2. **Hyperparameter Optimization**: Grid search for optimal parameters
3. **Additional Algorithms**: SVM, Neural Networks, Gradient Boosting
4. **Feature Selection**: Compare with univariate feature selection methods

### Computational Optimizations
1. **Parallel Processing**: Leverage multi-core systems
2. **GPU Acceleration**: CUDA implementation for kernel methods
3. **Memory Mapping**: Handle datasets larger than RAM
4. **Distributed Computing**: Scale to massive datasets

### Advanced Analysis
1. **Manifold Learning**: t-SNE, UMAP for nonlinear dimensionality reduction
2. **Autoencoders**: Neural network-based dimensionality reduction
3. **Feature Importance**: Analyze which original features matter most
4. **Reconstruction Error**: Evaluate information loss quantitatively

---

## Limitations and Considerations

### Memory Constraints
- **Kernel PCA**: Full dataset analysis limited by available memory
- **Subset Analysis**: Results based on 10,000 samples for Kernel PCA
- **Hardware Dependency**: Performance varies with system specifications

### Algorithm-Specific Findings
- **Random Forest Paradox**: Unexpected training time increase with PCA
- **Implementation Details**: Results may vary with different libraries/versions
- **Parameter Sensitivity**: Results depend on algorithm hyperparameters

### Generalizability
- **Dataset-Specific**: Results specific to MNIST characteristics
- **Domain Limitations**: May not generalize to all image classification tasks
- **Scale Considerations**: Behavior may differ with larger/smaller datasets

## Reproducibility

### Random Seed Management
```
# Ensure reproducible results
random_state = 42
np.random.seed(random_state)

# Consistent classifier initialization
rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
sgd = SGDClassifier(random_state=random_state)
```

### Environment Specifications
- **Python Version**: 3.8+
- **Scikit-learn**: 1.0+
- **NumPy**: 1.20+
- **Matplotlib**: 3.5+

---

## Conclusion

This comprehensive analysis demonstrates that **dimensionality reduction effectiveness is highly algorithm-dependent**. While PCA provides significant benefits for linear algorithms like SGDClassifier (73% training time reduction with minimal accuracy loss), it can paradoxically slow down tree-based algorithms like Random Forest. The project successfully shows that:

1. **Feature reduction works best with linear algorithms**
2. **Memory and storage benefits are universal**
3. **Kernel PCA provides excellent visualization capabilities**
4. **Algorithm selection must consider dimensionality reduction compatibility**

The insights gained from this analysis are valuable for designing efficient machine learning pipelines and making informed decisions about preprocessing strategies in real-world applications.

## References

1. Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
2. Schölkopf, B., Smola, A., & Müller, K. R. (1998). Nonlinear component analysis as a kernel eigenvalue problem. *Neural Computation*, 10(5), 1299-1319.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
4. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
5. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
