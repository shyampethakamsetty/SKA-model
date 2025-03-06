import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SKAModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, output_size=3):
        super(SKAModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Deeper architecture with batch normalization
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size//2)
        self.bn3 = nn.BatchNorm1d(hidden_size//2)
        self.layer4 = nn.Linear(hidden_size//2, output_size)
        
        # Initialize weights with improved initialization
        self._init_weights()
        
        # Knowledge tensors and history
        self.Z = None
        self.entropy_history = []
        self.best_accuracy = 0
        self.best_weights = None
        self.no_improvement_count = 0
        self.momentum = None
        self.velocity = {
            'layer1': torch.zeros_like(self.layer1.weight),
            'layer2': torch.zeros_like(self.layer2.weight),
            'layer3': torch.zeros_like(self.layer3.weight),
            'layer4': torch.zeros_like(self.layer4.weight)
        }
        
    def _init_weights(self):
        """Initialize weights with He initialization"""
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        h1 = F.relu(self.bn1(self.layer1(x)))
        h2 = F.relu(self.bn2(self.layer2(h1)))
        h3 = F.relu(self.bn3(self.layer3(h2)))
        out = self.layer4(h3)
        return out
    
    def compute_knowledge_tensor(self, x_batch, y_batch):
        """Compute knowledge tensor Z for given batch"""
        with torch.no_grad():
            h1 = F.relu(self.bn1(self.layer1(x_batch)))
            h2 = F.relu(self.bn2(self.layer2(h1)))
            h3 = F.relu(self.bn3(self.layer3(h2)))
            self.Z = self.layer4(h3)
            
            # Compute current accuracy
            _, predicted = torch.max(self.Z.data, 1)
            accuracy = (predicted == y_batch).sum().item() / len(y_batch)
            
            # Save best weights
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_weights = {
                    'layer1': self.layer1.weight.clone(),
                    'layer2': self.layer2.weight.clone(),
                    'layer3': self.layer3.weight.clone(),
                    'layer4': self.layer4.weight.clone()
                }
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                
            # If no improvement for a while, restore best weights
            if self.no_improvement_count > 30:
                self.restore_best_weights()
                self.no_improvement_count = 0
            
            return self.Z
    
    def compute_entropy(self):
        """Compute entropy of knowledge tensor"""
        if self.Z is None:
            return 0
        
        probs = F.softmax(self.Z, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=1).mean()
        return entropy.item()
    
    def update_weights(self, entropy_reduction_target=0.1, learning_rate=0.01, momentum=0.9):
        """Update weights with momentum and adaptive learning"""
        if self.Z is None:
            return
        
        initial_entropy = self.compute_entropy()
        
        # Update all layers with momentum
        with torch.no_grad():
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
            
            # Generate updates with momentum
            for layer, name in zip(layers, layer_names):
                # Generate new delta
                delta = torch.randn_like(layer.weight) * learning_rate
                
                # Apply momentum
                self.velocity[name] = momentum * self.velocity[name] + (1 - momentum) * delta
                layer.weight += self.velocity[name]
            
            new_entropy = self.compute_entropy()
            
            # Adaptive reversion based on entropy change
            if new_entropy > initial_entropy:
                reversion_prob = min(0.95, new_entropy / initial_entropy)
                if torch.rand(1).item() < reversion_prob:
                    for layer, name in zip(layers, layer_names):
                        layer.weight -= self.velocity[name]
                        self.velocity[name] *= 0.5  # Reduce velocity on reversion
            
        self.entropy_history.append(initial_entropy)
    
    def restore_best_weights(self):
        """Restore the weights that gave the best accuracy"""
        if self.best_weights is not None:
            self.layer1.weight.data.copy_(self.best_weights['layer1'])
            self.layer2.weight.data.copy_(self.best_weights['layer2'])
            self.layer3.weight.data.copy_(self.best_weights['layer3'])
            self.layer4.weight.data.copy_(self.best_weights['layer4'])
