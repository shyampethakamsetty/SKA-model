# Structured Knowledge Acquisition (SKA) Model Implementation for Iris Dataset

## Project Overview
This project implements the Structured Knowledge Acquisition (SKA) model, an alternative neural network training approach that uses entropy-based learning instead of traditional backpropagation. The implementation is specifically adapted for the Iris dataset classification task.

## Dataset Information
The Iris dataset is a classic machine learning dataset that contains:
- 150 samples of iris flowers
- 4 features (sepal length, sepal width, petal length, petal width)
- 3 classes (setosa, versicolor, virginica)
- Each class has 50 samples
- Dataset is loaded from scikit-learn's built-in datasets

## Project Structure

### Core Files
1. `ska_model.py`
   - Core implementation of the SKA model
   - Features:
     - Batch normalization layers
     - Deep architecture (4 layers)
     - He weight initialization
     - Momentum-based updates
     - Adaptive learning mechanisms
     - Entropy-based weight updates
     - Best weights tracking and restoration

2. `train_iris_ska.py`
   - Training script and evaluation pipeline
   - Implements:
     - Data preprocessing and feature engineering
     - Model training loops
     - Performance evaluation
     - Comparison with traditional neural network
     - Visualization of results
     - Detailed progress monitoring

3. `requirements.txt`
   - Project dependencies:
     - numpy: Numerical computations
     - scikit-learn: Dataset and metrics
     - pytorch: Deep learning framework
     - matplotlib: Visualization
     - seaborn: Enhanced visualizations
     - pandas: Data manipulation

4. `README.md`
   - Quick start guide
   - Setup instructions
   - Basic usage examples

## Technical Details

### SKA Model Architecture
- Input Layer: Accepts engineered features (original + interaction terms + polynomial features)
- Hidden Layers:
  - Layer 1: 32 neurons with BatchNorm and ReLU
  - Layer 2: 32 neurons with BatchNorm and ReLU
  - Layer 3: 16 neurons with BatchNorm and ReLU
- Output Layer: 3 neurons (one for each iris class)

### Training Approach
1. **SKA Learning**:
   - Uses entropy reduction as learning signal
   - No backpropagation required
   - Adaptive weight updates based on entropy changes
   - Momentum-based learning with velocity tracking
   - Dynamic learning rate scheduling

2. **Feature Engineering**:
   - Original features (4)
   - Interaction terms between features
   - Polynomial features (squared terms)
   - All features are standardized

### Performance Monitoring
- Training accuracy tracking
- Entropy evolution monitoring
- Early stopping with patience
- Best model weights saving
- Confusion matrix analysis
- Detailed classification reports

## Why SKA?

1. **Theoretical Advantages**:
   - No gradient calculations required
   - Can potentially escape local minima better
   - More biologically plausible learning mechanism
   - Simpler weight update rules

2. **Practical Benefits**:
   - Memory efficient (no gradient storage)
   - Can work with discrete activations
   - Potentially faster training in some scenarios
   - More interpretable learning process through entropy

## Results Visualization
The training process generates:
1. Accuracy comparison plots between SKA and traditional NN
2. Entropy evolution plots
3. Confusion matrices
4. Detailed classification reports

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run training:
   ```bash
   python train_iris_ska.py
   ```

3. View results:
   - Check console output for metrics
   - View `training_results.png` for plots
   - Analyze confusion matrices and classification reports

## Future Improvements
1. Hyperparameter optimization
2. Cross-validation support
3. Support for different datasets
4. More sophisticated feature engineering
5. Ensemble methods with traditional models

## References
- Iris Dataset: Fisher, R.A. (1936)
- SKA Methodology: Based on entropy-driven learning principles
- Implementation: Adapted from image classification version for tabular data
