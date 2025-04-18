"""Random Forest implementation with differential privacy for federated learning."""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from collections import OrderedDict
from typing import List, Tuple, Dict, Any
import joblib

class RFModel:
    """Random Forest wrapper class for compatibility with federated learning."""
    
    def __init__(self, n_estimators=100, max_depth=10):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.is_fitted = False
    
    def get_params(self) -> List[np.ndarray]:
        """Extract model parameters for federated learning."""
        if not self.is_fitted:
            return None
            
        params = []
        for tree in self.model.estimators_:
            # Extract tree structure
            tree_params = {
                'n_features': tree.tree_.n_features,
                'n_classes': tree.tree_.n_classes,
                'n_outputs': tree.tree_.n_outputs,
                'max_depth': tree.get_depth(),
                'node_count': tree.tree_.node_count,
                'nodes': {
                    'left_child': tree.tree_.children_left.copy(),
                    'right_child': tree.tree_.children_right.copy(),
                    'feature': tree.tree_.feature.copy(),
                    'threshold': tree.tree_.threshold.copy(),
                    'impurity': tree.tree_.impurity.copy(),
                    'n_node_samples': tree.tree_.n_node_samples.copy(),
                    'weighted_n_node_samples': tree.tree_.weighted_n_node_samples.copy(),
                    'value': tree.tree_.value.copy()
                }
            }
            params.append(tree_params)
            
        return params

    def set_params(self, parameters: List[Dict[str, Any]]) -> None:
        """Set model parameters from federated learning updates."""
        if parameters is None:
            return
            
        # Create new trees with the received parameters
        self.model.estimators_ = []
        for tree_params in parameters:
            tree = RandomForestClassifier(
                n_estimators=1,
                max_depth=tree_params['max_depth']
            ).fit(np.zeros((2, tree_params['n_features'])), [0, 1])
            
            tree = tree.estimators_[0]
            tree.tree_.n_features = tree_params['n_features']
            tree.tree_.n_classes = tree_params['n_classes']
            tree.tree_.n_outputs = tree_params['n_outputs']
            
            # Set tree structure
            nodes = tree_params['nodes']
            tree.tree_.children_left = nodes['left_child']
            tree.tree_.children_right = nodes['right_child']
            tree.tree_.feature = nodes['feature']
            tree.tree_.threshold = nodes['threshold']
            tree.tree_.impurity = nodes['impurity']
            tree.tree_.n_node_samples = nodes['n_node_samples']
            tree.tree_.weighted_n_node_samples = nodes['weighted_n_node_samples']
            tree.tree_.value = nodes['value']
            
            self.model.estimators_.append(tree)
            
        self.is_fitted = True

    def add_noise(self, epsilon: float, delta: float) -> None:
        """Add differential privacy noise to the model."""
        if not self.is_fitted:
            return
            
        sensitivity = 1.0  # Assuming normalized data
        noise_scale = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        for tree in self.model.estimators_:
            # Add noise to node thresholds
            noise = np.random.laplace(0, sensitivity * noise_scale, 
                                    size=tree.tree_.threshold.shape)
            tree.tree_.threshold += noise
            
            # Add noise to node values (class predictions)
            noise = np.random.laplace(0, sensitivity * noise_scale, 
                                    size=tree.tree_.value.shape)
            tree.tree_.value += noise

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the random forest model."""
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the random forest model."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions using the random forest model."""
        return self.model.predict_proba(X)

def get_model_weights(model: RFModel) -> List[np.ndarray]:
    """Get model parameters for federated learning."""
    return model.get_params()

def set_model_weights(model: RFModel, parameters: List[np.ndarray]) -> None:
    """Set model parameters from federated learning updates."""
    model.set_params(parameters)

def load_data(partition_id: int, num_partitions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and partition the dataset."""
    # Load your dataset here
    df = pd.read_csv("C:/Users/kouek/OneDrive - Université de Moncton/Documents/FL_IoMT/opacus/opacus_fl/iomtTDmul5.csv")
    features = df.drop(columns=["attack_type"]).values
    labels = df["attack_type"].astype('category').cat.codes.values

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Create IID partition
    data_size = len(features)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    # Split data for this partition
    partition_size = data_size // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else data_size
    partition_indices = indices[start_idx:end_idx]

    partition_features = features[partition_indices]
    partition_labels = labels[partition_indices]

    # Split into train and test
    return train_test_split(
        partition_features, 
        partition_labels, 
        test_size=0.2, 
        random_state=42
    )

def evaluate_model(model: RFModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    predictions = model.predict(X_test)
    
    return {
        "accuracy": (predictions == y_test).mean(),
        "precision": precision_score(y_test, predictions, average="weighted"),
        "recall": recall_score(y_test, predictions, average="weighted"),
        "f1_score": f1_score(y_test, predictions, average="weighted")
    }