import os
import json
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Import architectures directly from your repository modules
from model import get_pipeline_model

def run_fashion_testing():
    print("--- Starting Automated Fashion-MNIST Testing Harness ---", flush=True)
    
    # 1. Configuration Resolution Pathing
    config_path = os.getenv("TRAINING_CONFIG_PATH", "/app/configs/training_config.yaml")
    if not os.path.exists(config_path):
        config_path = "D:/Project/training_config.yaml"  # Local fallback context
        
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
        
    dataset_name = "fashion_mnist"
    cfg = raw_config["datasets"][dataset_name]
    checkpoint_dir = Path(raw_config.get("checkpoint_dir", "/app/data/checkpoints"))
    data_dir = raw_config.get("data_dir", "/app/data")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Build and Unpack Model Checkpoint
    model = get_pipeline_model(
        architecture=cfg["model"]["architecture"],
        dataset_name=dataset_name,
        num_classes=int(cfg["model"]["num_classes"]),
        use_se=bool(cfg["model"]["use_se"])
    )
    
    model_path = checkpoint_dir / cfg["output"]["model_name"]
    if not model_path.exists():
        model_path = Path(cfg["output"]["model_name"])  # Local root boundary fallback
        
    print(f"Loading weights from: {model_path}", flush=True)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    # 3. Data Transformations & Labels Configuration
    test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Pad(2),  # Up-samples 28x28 grayscale matrices to 32x32 for SEResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530])
    ])
    
    fashion_labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True)
    
    # 4. Processing Execution and Plot Construction
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    np.random.seed(42)  # Seeds random metrics generation for testing consistency
    
    for i, ax in enumerate(axes.flatten()):
        img_no = np.random.randint(0, len(test_set))
        raw_img, label_idx = test_set[img_no]
        
        # Convert and prepare image for in-memory model evaluation pass
        img = raw_img.convert("L")
        input_tensor = test_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
        pred_idx = predicted.item()
        conf = probabilities[0][pred_idx].item() * 100
        
        true_name = fashion_labels[label_idx]
        pred_name = fashion_labels[pred_idx]
        
        # Render item plot data context
        ax.imshow(raw_img, cmap='gray')
        # Highlights prediction titles in green if correct, or red if misclassified
        title_color = "green" if pred_idx == label_idx else "red"
        ax.set_title(f"ID: {img_no}\nTrue: {true_name}\nPred: {pred_name}\n({conf:.1f}%)", 
                     fontsize=9, color=title_color)
        ax.axis('off')
        
        # Log clean structured results mapping lines directly onto standard output stream
        log_entry = {
            "sample_id": img_no,
            "true_label": true_name,
            "predicted_label": pred_name,
            "confidence": round(conf, 2)
        }
        print(json.dumps(log_entry), flush=True)
        
    # 5. Persistent Artifact Target Dumps
    output_dir = Path("/app/data/prediction_results")
    if not output_dir.exists():
        output_dir = Path("data/prediction_results")  # Local workspace context safety
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / "fashion_mnist_results.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"\n[SUCCESS] Verification visual saved persistently to: {save_path}", flush=True)

if __name__ == "__main__":
    run_fashion_testing()
