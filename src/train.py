# main.py
import os
import json
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import local module functions
from dataset import get_dataloaders
from model import get_pipeline_model

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return total_loss / total, correct / total

@torch.no_grad()  # Secure decorator ensuring no gradient caching during validation testing
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return total_loss / total, correct / total

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Config routing
    config_path = Path("/app/configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path(r"D:/Project/training_config.yaml")
        print(config_path)
    full_config = load_config(str(config_path))
    dataset_name = full_config["active_dataset"] 
    config = full_config["datasets"][dataset_name]
    data_dir = full_config["data_dir"]
    checkpoint_dir = Path(full_config["checkpoint_dir"])
    
    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model construction using the custom CNN factory settings
    model = get_pipeline_model(
        architecture=config["model"]["architecture"],
        dataset_name=dataset_name,
        num_classes=config["model"]["num_classes"],
        in_channels=config["model"]["in_channels"],
        use_se=config["model"]["use_se"]
    ).to(device)
    
    # 4. Fetch data loaders dynamically
    train_loader, val_loader = get_dataloaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=config["training"]["batch_size"]
    )
    
    # 5. Optimization tracking configurations
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]
    #checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 6. Training iteration execution
    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        print(json.dumps(log_entry), flush=True)
        
        # Checkpoint evaluation check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / config["output"]["model_name"]
            
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, save_path)
            
            print(json.dumps({"event": "checkpoint_saved", "path": str(save_path)}), flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({"event": "early_stopping", "epoch": epoch + 1}), flush=True)
                break
                
    print(json.dumps({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)}), flush=True)

if __name__ == "__main__":
    main()
