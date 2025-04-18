"""Script to visualize training results and evaluate model on training data."""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd

class LegacyNODEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.internal_nodes = nn.Parameter(torch.zeros(96))
        self.internal_bias = nn.Parameter(torch.zeros(34))
        self.leaves = nn.Parameter(torch.zeros(160))
        self.feature_mask = nn.Parameter(torch.zeros(64))

    def forward(self, x):
        # Add numerical stability checks
        x = torch.clamp(x, -10, 10)  # Prevent extreme values
        
        # Ensure input is the right size with stable pooling
        x = F.adaptive_avg_pool1d(x.unsqueeze(1), 64).squeeze(1)
        
        # Apply feature mask with numerical stability
        feature_mask = torch.clamp(self.feature_mask, -10, 10)
        feature_mask = feature_mask.view(1, -1)
        x = x * torch.sigmoid(feature_mask)
        
        # Add small epsilon to prevent complete zeros
        x = x + 1e-8
        return x

class LegacyNODEModel(nn.Module):
    def __init__(self, in_features=24, num_classes=6):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = 64
        
        # Parameters with correct sizes
        self.input_weight = nn.Parameter(torch.zeros(800))  # Will be reshaped to (32, 25)
        self.input_bias = nn.Parameter(torch.zeros(64))
        self.batch_norm1 = nn.BatchNorm1d(64)
        
        # NODE block
        self.node_block = LegacyNODEBlock()
        
        # Classifier parameters
        self.classifier_weight1 = nn.Parameter(torch.zeros(1056))  # Will be reshaped to (16, 66)
        self.classifier_bias1 = nn.Parameter(torch.zeros(64))
        self.batch_norm2 = nn.BatchNorm1d(64)
        
        # Final layer parameters
        self.final_weight = nn.Parameter(torch.zeros(224))  # Will be reshaped to (6, 37)
        self.final_bias = nn.Parameter(torch.zeros(6))

    def forward(self, x):
        if not self.training:
            print("\nInput shape:", x.shape)
        
        # First layer transformation
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # Flatten input
        
        # Reshape input weight for first layer
        weight_matrix = self.input_weight[:self.in_features * 32].view(32, self.in_features)
        x = F.linear(x_flat, weight_matrix, self.input_bias[:32])
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        if not self.training:
            print("After input transform:", x.shape)
        
        # NODE block
        x = self.node_block(x)
        
        if not self.training:
            print("After NODE block:", x.shape)
        
        # Classifier
        x_size = x.size(1)
        weight_matrix = self.classifier_weight1[:x_size * 64].view(64, x_size)
        x = F.linear(x, weight_matrix, self.classifier_bias1)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Final layer
        x_size = x.size(1)
        weight_matrix = self.final_weight[:x_size * 6].view(6, x_size)
        x = F.linear(x, weight_matrix, self.final_bias)
        
        if not self.training:
            print("Final output:", x.shape)
            print("Output logits:", x[0])  # Print first sample's logits
        
        return x

def generate_confusion_matrix(y_true, y_pred, class_names):
    """Generate and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

class ModelInference:
    def __init__(self, model_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LegacyNODEModel().to(self.device)
        self.load_model_weights(model_weights_path)
        self.labels = ["Benign", "Dos", "Scanning", "MQTT", "CAM_Overflow", "Spoofing"]

    def load_model_weights(self, weights_path):
        """Load trained model weights."""
        try:
            weights = np.load(weights_path, allow_pickle=True)
            state_dict = {}
            
            def convert_weight(w):
                if isinstance(w, np.ndarray) and w.dtype.kind == 'S':
                    w = np.frombuffer(w.tobytes(), dtype=np.float32)
                # Clip values to prevent numerical instability
                return np.clip(w, -10, 10)

            print("\nLoading weights...")
            for key in weights.files:
                w = convert_weight(weights[key])
                print(f"{key}: shape={w.shape}, size={w.size}, "
                    f"min={w.min():.4f}, max={w.max():.4f}")
            
            # Load weights with exact sizes
            state_dict['input_weight'] = torch.from_numpy(convert_weight(weights['arr_0'])).float()
            state_dict['input_bias'] = torch.from_numpy(convert_weight(weights['arr_1'])).float()
            
            # BatchNorm1 parameters
            state_dict['batch_norm1.weight'] = torch.from_numpy(convert_weight(weights['arr_2'])).float()
            state_dict['batch_norm1.bias'] = torch.from_numpy(convert_weight(weights['arr_3'])).float()
            state_dict['batch_norm1.running_mean'] = torch.zeros(64)
            state_dict['batch_norm1.running_var'] = torch.ones(64)
            
            # NODE block parameters
            state_dict['node_block.internal_nodes'] = torch.from_numpy(convert_weight(weights['arr_4'])).float()
            state_dict['node_block.internal_bias'] = torch.from_numpy(convert_weight(weights['arr_5'])).float()
            state_dict['node_block.leaves'] = torch.from_numpy(convert_weight(weights['arr_6'])).float()
            state_dict['node_block.feature_mask'] = torch.from_numpy(convert_weight(weights['arr_7'])).float()
            
            # Classifier parameters
            state_dict['classifier_weight1'] = torch.from_numpy(convert_weight(weights['arr_10'])).float()
            state_dict['classifier_bias1'] = torch.from_numpy(convert_weight(weights['arr_11'])).float()
            
            # BatchNorm2 parameters
            state_dict['batch_norm2.weight'] = torch.from_numpy(convert_weight(weights['arr_12'])).float()
            state_dict['batch_norm2.bias'] = torch.from_numpy(convert_weight(weights['arr_13'])).float()
            state_dict['batch_norm2.running_mean'] = torch.zeros(64)
            state_dict['batch_norm2.running_var'] = torch.ones(64)
            
            # Final layer parameters
            state_dict['final_weight'] = torch.from_numpy(convert_weight(weights['arr_14'])).float()
            state_dict['final_bias'] = torch.from_numpy(convert_weight(weights['arr_15'])[:6]).float()
            
            # Load state dict
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("\nModel weights loaded successfully")
            
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            raise

    def predict(self, features):
        """Make predictions on new data."""
        self.model.eval()
        predictions = []
        probabilities = []
        total_batches = (len(features) + 63) // 64
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            batch_size = 64
            
            for i in range(0, len(features), batch_size):
                batch = features_tensor[i:i + batch_size]
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                
                if i == 0:
                    print("\nSample of model outputs:")
                    print("Raw outputs (first batch):")
                    print(outputs[0])
                    print("\nProbabilities (first batch):")
                    print(probs[0])
                
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                probabilities.append(probs.cpu().numpy())
                
                if (i // batch_size) % 100 == 0:
                    print(f"Processed batch {i // batch_size + 1}/{total_batches}")
        
        return np.array(predictions)

    def evaluate_dataset(self, data_path, output_path='inference_results.txt'):
        """Evaluate model on dataset."""
        print("Starting evaluation on dataset...")
        features, true_labels = self.preprocess_data(data_path)
        
        print("\nMaking predictions...")
        predictions = self.predict(features)
        
        print("\nGenerating evaluation report...")
        with open(output_path, 'w') as f:
            f.write("Model Evaluation Results\n")
            f.write("=======================\n\n")
            
            report = classification_report(
                true_labels, 
                predictions,
                target_names=self.labels,
                digits=4
            )
            f.write("Classification Report:\n")
            f.write(report)
            print("\nClassification Report:")
            print(report)
            
            generate_confusion_matrix(true_labels, predictions, self.labels)
        
        print(f"\nEvaluation completed. Results saved to {output_path}")

    def preprocess_data(self, data_path):
        """Preprocess data using original labels."""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        print("\nUnique labels in dataset:", df["attack_type"].unique())
        print("\nLabel distribution:")
        print(df["attack_type"].value_counts())
        
        features = df.drop(columns=["attack_type"]).values
        labels = df["attack_type"].astype('category').cat.codes.values
        
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        print(f"\nPreprocessed {len(features)} samples")
        
        return features, labels

def main():
    print("Starting visualization and inference process...")
    
    try:
        # Add this to your main function temporarily
        weights = np.load('C:/Users/kouek/OneDrive - Université de Moncton/Documents/FL_IoMT/opacus/round-10-weights.npz', allow_pickle=True)
        for key in weights.files:
            print(f"\n{key}:")
            print(f"Shape: {weights[key].shape}")
            print(f"Size: {weights[key].size}")
        
        model_path = 'C:/Users/kouek/OneDrive - Université de Moncton/Documents/FL_IoMT/opacus/round-5-weights.npz'
        print(f"\nLoading model from {model_path}")
        model_inference = ModelInference(model_path)
        
        training_data_path = "C:/Users/kouek/OneDrive - Université de Moncton/Documents/FL_IoMT/opacus/opacus_fl/iomtTDmul5.csv"
        print(f"\nEvaluating model on dataset: {training_data_path}")
        model_inference.evaluate_dataset(training_data_path)
        
    except Exception as e:
        print(f"\nError during model inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()