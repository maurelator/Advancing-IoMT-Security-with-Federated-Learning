"""Flower server implementation for Random Forest with differential privacy."""

import logging
from typing import List, Tuple, Dict
import json
import numpy as np
from flwr.common import Context, Metrics, Parameters, NDArrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

logging.getLogger("flwr").setLevel(logging.INFO)

# Initialize lists for metrics evolution
training_losses: List[float] = []
accuracies: List[float] = []
precisions: List[float] = []
recalls: List[float] = []
f1_scores: List[float] = []

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate weighted average of evaluation metrics."""
    # Extract metrics with their weights (number of examples)
    weighted_metrics = {
        "accuracy": [num_examples * m["accuracy"] for num_examples, m in metrics],
        "precision": [num_examples * m["precision"] for num_examples, m in metrics],
        "recall": [num_examples * m["recall"] for num_examples, m in metrics],
        "f1_score": [num_examples * m["f1_score"] for num_examples, m in metrics]
    }
    
    # Calculate total number of examples
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    # Calculate weighted averages
    return {
        metric: sum(values) / total_examples 
        for metric, values in weighted_metrics.items()
    }

def aggregate_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training metrics and save them."""
    global accuracies, precisions, recalls, f1_scores
    
    # Calculate weighted averages
    agg_metrics = weighted_average(metrics)
    
    # Store metrics for plotting
    accuracies.append(agg_metrics["accuracy"])
    precisions.append(agg_metrics["precision"])
    recalls.append(agg_metrics["recall"])
    f1_scores.append(agg_metrics["f1_score"])
    
    # Save metrics to JSON file
    metrics_dict = {
        "accuracies": accuracies,
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
    }
    
    with open("rf_metrics.json", "w") as f:
        json.dump(metrics_dict, f)
    
    return agg_metrics

class RFStrategy(FedAvg):
    """Custom FedAvg strategy for Random Forest models."""
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[NDArrays, Dict]],
        failures: List[BaseException],
    ) -> Tuple[NDArrays, Dict]:
        """Aggregate model updates from clients."""
        if not results:
            return None, {}
            
        # Extract trained parameters from each client
        client_params = [parameters for parameters, _ in results]
        num_clients = len(client_params)
        
        # Initialize aggregated parameters with the structure of the first client
        aggregated_trees = []
        num_trees = len(client_params[0])
        
        # Aggregate each tree from all clients
        for tree_idx in range(num_trees):
            # Initialize aggregated tree with structure from first client
            agg_tree = client_params[0][tree_idx].copy()
            
            # Average the numerical parameters across clients
            for param_key in ['threshold', 'impurity', 'value']:
                if param_key in agg_tree['nodes']:
                    param_sum = np.zeros_like(agg_tree['nodes'][param_key])
                    for client_idx in range(num_clients):
                        param_sum += client_params[client_idx][tree_idx]['nodes'][param_key]
                    agg_tree['nodes'][param_key] = param_sum / num_clients
            
            # Keep structural parameters from the first client
            # (tree structure, feature indices, etc.)
            aggregated_trees.append(agg_tree)
        
        # Save model for the final round
        if server_round == 5:  # Adjust based on your total rounds
            with open(f"round-{server_round}-rf-model.json", "w") as f:
                json.dump(aggregated_trees, f)
        
        # Get metrics
        metrics = aggregate_fit_metrics([
            (num_examples, fit_metrics)
            for _, (num_examples, fit_metrics) in results
        ])
        
        return aggregated_trees, metrics

def server_fn(context: Context) -> ServerAppComponents:
    """Initialize and configure the server."""
    # Get number of rounds from context
    num_rounds = context.run_config["num-server-rounds"]
    
    # Initialize empty model parameters
    initial_parameters = []  # Will be populated by first client
    
    # Create custom strategy
    strategy = RFStrategy(
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        initial_parameters=initial_parameters,
    )
    
    # Create server configuration
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(config=config, strategy=strategy)

# Initialize server application
app = ServerApp(server_fn=server_fn)

def run_server():
    """Run the server application."""
    try:
        app.start()
    except Exception as e:
        print(f"Error during server execution: {str(e)}")

if __name__ == "__main__":
    run_server()