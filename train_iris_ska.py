import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ska_model import SKAModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import torch.nn as nn
import torch.optim as optim

def load_and_preprocess_data():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Add feature engineering
    X = np.hstack([
        X,
        X[:, 0:2] * X[:, 2:4],  # Interaction terms
        X ** 2  # Polynomial features
    ])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return X_train, X_test, y_train, y_test, iris.target_names

def train_ska_model(X_train, y_train, epochs=150, K=30):
    model = SKAModel(input_size=X_train.shape[1])  # Adjust for engineered features
    accuracies = []
    learning_rate = 0.03  # Increased initial learning rate
    
    best_accuracy = 0
    patience = 25
    no_improvement = 0
    min_entropy = float('inf')
    
    for epoch in range(epochs):
        # Adaptive K steps
        K_steps = max(15, K - epoch // 5)  # More gradual K reduction
        
        # Adaptive learning rate
        current_lr = learning_rate * (0.97 ** (epoch // 5))
        
        for _ in range(K_steps):
            model.compute_knowledge_tensor(X_train, y_train)
            model.update_weights(learning_rate=current_lr, momentum=0.9)
            
            # Track minimum entropy
            current_entropy = model.entropy_history[-1]
            min_entropy = min(min_entropy, current_entropy)
        
        # Compute accuracy
        with torch.no_grad():
            outputs = model(X_train)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_train).sum().item() / len(y_train)
            accuracies.append(accuracy)
            
            # Early stopping with improved criteria
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                no_improvement = 0
            elif accuracy < best_accuracy - 0.05:  # Allow small fluctuations
                no_improvement += 1
            else:
                no_improvement = max(0, no_improvement - 1)  # Reduce counter for stable performance
                
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Accuracy: {accuracy:.4f}, '
                  f'Entropy: {current_entropy:.4f}, K: {K_steps}, LR: {current_lr:.4f}')
    
    # Restore best weights before returning
    model.restore_best_weights()
    return model, accuracies

def train_traditional_nn(X_train, y_train, epochs=100):
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),  # Larger network for engineered features
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.2),
        nn.Linear(32, 3)
    )
    
    # Initialize weights
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    accuracies = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        model.eval()
        with torch.no_grad():
            outputs = model(X_train)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_train).sum().item() / len(y_train)
            accuracies.append(accuracy)
            scheduler.step(accuracy)
            
        if (epoch + 1) % 10 == 0:
            print(f'Traditional NN - Epoch [{epoch+1}/{epochs}], '
                  f'Accuracy: {accuracy:.4f}, Loss: {loss.item():.4f}')
    
    return model, accuracies

def plot_results(ska_accuracies, trad_accuracies, entropy_history):
    plt.figure(figsize=(15, 5))
    
    # Plot accuracies
    plt.subplot(1, 2, 1)
    plt.plot(ska_accuracies, label='SKA Model', linewidth=2)
    plt.plot(trad_accuracies, label='Traditional NN', linewidth=2)
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot entropy
    plt.subplot(1, 2, 2)
    plt.plot(entropy_history, linewidth=2)
    plt.title('SKA Model Entropy Evolution')
    plt.xlabel('Update Step')
    plt.ylabel('Entropy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, target_names = load_and_preprocess_data()
    
    # Train SKA model
    print("\nTraining SKA Model...")
    ska_model, ska_accuracies = train_ska_model(X_train, y_train)
    
    # Train traditional neural network
    print("\nTraining Traditional Neural Network...")
    trad_model, trad_accuracies = train_traditional_nn(X_train, y_train)
    
    # Plot results
    plot_results(ska_accuracies, trad_accuracies, ska_model.entropy_history)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print("----------------")
    
    with torch.no_grad():
        # SKA Model evaluation
        ska_outputs = ska_model(X_test)
        _, ska_predicted = torch.max(ska_outputs.data, 1)
        ska_accuracy = accuracy_score(y_test, ska_predicted)
        
        print("\nSKA Model Results:")
        print(f"Test Accuracy: {ska_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, ska_predicted, target_names=target_names))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, ska_predicted))
        
        # Traditional NN evaluation
        trad_outputs = trad_model(X_test)
        _, trad_predicted = torch.max(trad_outputs.data, 1)
        trad_accuracy = accuracy_score(y_test, trad_predicted)
        
        print("\nTraditional NN Results:")
        print(f"Test Accuracy: {trad_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, trad_predicted, target_names=target_names))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, trad_predicted))

if __name__ == "__main__":
    main()
