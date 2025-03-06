# SKA Model Implementation for Iris Dataset

This project implements the Structured Knowledge Acquisition (SKA) model for the Iris dataset classification task. The implementation includes both the SKA approach and a traditional neural network for comparison.

## Project Structure
- `ska_model.py`: Core SKA model implementation
- `train_iris_ska.py`: Training script with data preparation and evaluation
- `requirements.txt`: Required Python packages
- `SKA_Iris_Implementation.ipynb`: Interactive Jupyter notebook with implementation details
- `description.md`: Comprehensive project documentation
- `report.md`: Detailed comparison of SKA vs traditional approaches

## Setup and Installation
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

### Option 1: Python Script
To train and evaluate both models using the Python script:
```bash
python train_iris_ska.py
```

### Option 2: Jupyter Notebook (Recommended for Learning)
To explore the implementation interactively:
1. Start Jupyter notebook server:
```bash
jupyter notebook
```
2. Open `SKA_Iris_Implementation.ipynb` in your browser
3. Run cells sequentially to understand the implementation

The notebook provides:
- Detailed explanations of each step
- Interactive visualizations
- Real-time model performance analysis
- Code execution with immediate feedback

This will:
1. Load and preprocess the Iris dataset
2. Train both SKA and traditional models
3. Generate comparison visualizations
4. Output detailed performance metrics

## Implementation Details
- Input features: 4 base + 8 engineered features
- SKA model architecture: 12 → 32 → 32 → 16 → 3
- Training parameters:
  - Epochs: 150
  - Initial K-steps: 30
  - Adaptive learning rate
  - Momentum-based updates
  - Batch normalization

## Output
The implementation generates:
1. Training progress updates in the console
2. Accuracy comparison plots
3. Entropy evolution visualization
4. Confusion matrices
5. Detailed classification reports

## Documentation
- `description.md`: Detailed project overview and technical details
- `report.md`: Comparative analysis of SKA vs traditional approaches
- Notebook comments and markdown cells: Step-by-step explanation
