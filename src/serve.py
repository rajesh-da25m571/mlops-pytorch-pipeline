import os
import io
import yaml
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from model import get_pipeline_model

app = FastAPI(title="Multi-Model Dynamic Inference Service Engine")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables updated dynamically at container boot
model = None
model_type = "fashion_mnist"

# FIXED: Explicit label arrays mapped to your operational profiles
LABELS_MAP = {
    "fashion_mnist": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ],
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck"
    ]
}

@app.on_event("startup")
def configure_inference_runtime():
    global model, model_type
    
    # 1. FIXED: Point directly to your single-source-of-truth configuration file
    config_path = os.getenv("CONFIG_PATH", "/app/configs/training_config.yaml")
    if not os.path.exists(config_path):
        config_path = "D:/Project/training_config.yaml"  # Local fallback directory
        
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
        
    # 2. DYNAMIC ROUTING: Determine model target from env variable or fallback yaml key
    model_type = os.getenv("ACTIVE_MODEL", raw_config.get("active_dataset", "fashion_mnist")).lower()
    
    if model_type not in LABELS_MAP:
        raise RuntimeError(f"Unsupported active serving model token configuration: {model_type}")
        
    # Extract the configuration parameters for the selected dataset
    dataset_cfg = raw_config["datasets"][model_type]
    
    print(f"Initializing container runtime infrastructure for model: {model_type.upper()}")
    
    # 3. FIXED: Pass config block data downstream to instantiate the correct input dimensions
    model = get_pipeline_model(
        architecture=dataset_cfg["model"]["architecture"],
        dataset_name=model_type,
        num_classes=int(dataset_cfg["model"]["num_classes"]),
        use_se=bool(dataset_cfg["model"]["use_se"])
    )
    
    # Locate weight path
    checkpoint_dir = raw_config.get("checkpoint_dir", "/app/checkpoints")
    model_name = dataset_cfg["output"]["model_name"]
    weights_path = os.path.join(checkpoint_dir, model_name)
    
    if not os.path.exists(weights_path):
        # Fallback to current working data context if root absolute directory path is missing
        weights_path = os.path.join("/app/checkpoints", model_name)
    
    if os.path.exists(weights_path):
        print(f"Loading serialized matrix weights from checkpoint file: {weights_path}")
        raw_checkpoint = torch.load(weights_path, map_location=device)
        
        # Safely unpack 'model_state_dict' from the dictionary payload structure
        if isinstance(raw_checkpoint, dict) and "model_state_dict" in raw_checkpoint:
            model.load_state_dict(raw_checkpoint["model_state_dict"])
        else:
            model.load_state_dict(raw_checkpoint)
            
        print(f"Successfully mounted weight definitions onto architecture layers.")
    else:
        print(f"Warning: Checkpoint weights not found at {weights_path}. Running base weights.")
        
    model.to(device)
    model.eval()

@app.get("/health")
def readiness_probe():
    if model is not None:
        return {"status": "healthy", "model_active": model_type}
    raise HTTPException(status_code=500, detail="Server model uninitialized")

@app.post("/predict")
async def process_inference(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model runtime context missing")
        
    try:
        img = Image.open(io.BytesIO(await file.read()))
        
        # Apply preprocessing transforms explicitly matching the active dataset profile
        if model_type == "fashion_mnist":
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Pad(2),  # Pad 28x28 grayscale to 32x32 to match residual architecture layers
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2860], std=[0.3530])
            ])
            img = img.convert("L")
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ])
            img = img.convert("RGB")
            
        tensor = transform(img).unsqueeze(0).to(device)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload image formatting")

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

    labels = LABELS_MAP[model_type]
    idx = predicted.item()
    confidence = probabilities[0][idx].item() * 100
    prediction = labels[idx]
    class_probs = {labels[i]: probabilities[0][i].item() for i in range(len(labels))}
    return {
        "active_model": model_type,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {k: round(v * 100, 2) for k, v in class_probs.items()}
        #"probabilities": {k: round(v * 100, 2) for k, v in class_probs.items()}
    }
