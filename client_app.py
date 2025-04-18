"""Flower client implementation for Random Forest with differential privacy."""

from typing import Dict, List, Tuple, Any
from flwr.common import Context, NDArrays, Parameters
from flwr.client import NumPyClient, ClientApp
import numpy as np
from .task import RFModel, get_model_weights, set_model_weights, load_data, evaluate_model

class RFFlowerClient(NumPyClient):
    """Flower client implementing Random Forest training."""

    def __init__(
        self,
        model: RFModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epsilon: float = 1.0,
        delta: float = 1e-5
    ) -> None:
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epsilon = epsilon
        self.delta = delta

    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Train the model on the local dataset."""
        # Update model if parameters are provided
        if parameters is not None:
            set_model_weights(self.model, parameters)

        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Add differential privacy noise
        self.model.add_noise(self.epsilon, self.delta)
        
        # Evaluate training performance
        train_metrics = evaluate_model(self.model, self.X_train, self.y_train)
        
        # Get model parameters
        parameters = get_model_weights(self.model)
        
        return parameters, len(self.X_train), train_metrics

    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the model on the local test dataset."""
        # Update model if parameters are provided
        if parameters is not None:
            set_model_weights(self.model, parameters)
            
        # Evaluate model
        metrics = evaluate_model(self.model, self.X_test, self.y_test)
        
        return 0.0, len(self.X_test), metrics

def client_fn(context: Context) -> ClientApp:
    """Create and configure a Flower client."""
    # Get client-specific configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Load partitioned data
    X_train, X_test, y_train, y_test = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions
    )
    
    # Initialize model
    model = RFModel(
        n_estimators=100,  # Adjust based on your needs
        max_depth=10       # Adjust based on your needs
    )
    
    # Calculate privacy parameters based on partition
    epsilon = 2.0 if partition_id % 2 == 0 else 1.5
    delta = context.run_config.get("target-delta", 1e-5)
    
    # Create and return client
    return RFFlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epsilon=epsilon,
        delta=delta
    ).to_client()

# Initialize client application
app = ClientApp(client_fn=client_fn)