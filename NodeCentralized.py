import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from opacus import PrivacyEngine
from tqdm import tqdm
import logging
import wandb
from task import NODEModel, IoMTDataset

def load_centralized_data(batch_size=64):
    df = pd.read_csv("C:/Users/kouek/OneDrive - Université de Moncton/Documents/FL_IoMT/opacus/ciciomt2024mul5.csv")
    features = df.drop(columns=["attack_type"]).values
    labels = df["attack_type"].astype('category').cat.codes.values
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    train_dataset = IoMTDataset(X_train, y_train)
    test_dataset = IoMTDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

def train_centralized(
    epochs=10,
    learning_rate=0.001,
    target_epsilon=5.0,
    target_delta=1e-5,
    max_grad_norm=1.0,
    noise_multiplier=1.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NODEModel().to(device)
    
    # Initialize wandb
    wandb.init(project="node-centralized", config={
        "epochs": epochs,
        "learning_rate": learning_rate,
        "target_epsilon": target_epsilon,
        "noise_multiplier": noise_multiplier
    })
    
    train_loader, test_loader = load_centralized_data()
    
    # Setup privacy engine
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    privacy_engine = PrivacyEngine(secure_mode=False)
    
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm
    )
    
    # Calculate class weights for imbalanced dataset
    total_samples = len(train_loader.dataset)
    class_counts = torch.zeros(6)
    for batch in train_loader:
        labels = batch["label"]
        for i in range(6):
            class_counts[i] += (labels == i).sum()
    
    class_weights = total_samples / (6 * class_counts)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
        
        # Log metrics
        wandb.log({
            "train_loss": avg_loss,
            "test_loss": test_loss / len(test_loader),
            "accuracy": accuracy,
            "epsilon": epsilon
        })
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, "
              f"Test Accuracy = {accuracy:.2f}%, "
              f"ε = {epsilon:.2f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pt")
    
    wandb.finish()
    return model

def evaluate_model(model, test_loader, device):
    class_labels = ["Benign", "DDoS", "DoS", "Recon", "Spoofing", "MQTT"]
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = precision_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Generate and plot confusion matrix with class labels
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )
    disp.plot(cmap='Blues', values_format='d')
    plt.title('IoMT Attack Detection Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Calculate per-class metrics
    per_class_precision = precision_score(all_labels, all_preds, average=None)
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    
    # Print per-class metrics
    print("\nPer-class Metrics:")
    print("-" * 60)
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 60)
    for i, class_name in enumerate(class_labels):
        print(f"{class_name:<12} {per_class_precision[i]:>10.4f} {per_class_recall[i]:>10.4f} {per_class_f1[i]:>10.4f}")
    print("-" * 60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = train_centralized()
    
    # Final evaluation
    train_loader, test_loader = load_centralized_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(model, test_loader, device)
    
    print("\nFinal Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nConfusion matrix saved as 'confusion_matrix.png'")