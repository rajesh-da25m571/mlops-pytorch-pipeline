import os
import io
import yaml
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from model import get_pipeline_model

app = FastAPI(title="Multi-Model SE-ResNet Inference Service Engine")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
model_type = "cifar10"

LABELS_MAP = {
    "mnist": [str(i) for i in range(10)],
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
}

@app.on_event("startup")
def configure_inference_runtime():
    global model, model_type
    
    deploy_cfg_path = os.getenv("DEPLOY_CONFIG_PATH", "config/deploy_config.yaml")
    with open(deploy_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    model_type = os.getenv("ACTIVE_MODEL", cfg.get("active_serving_model", "cifar10")).lower()
    
    # Intialize your structural factory block architecture
    model = get_pipeline_model(dataset_name=model_type, num_classes=10)
    
    w_key = "mnist_weights_path" if model_type == "mnist" else "cifar10_weights_path"
    weights_path = cfg.get(w_key, f"data/{model_type}_best.pth")
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Successfully loaded {model_type} weights architecture from {weights_path}")
    else:
        print(f"Warning: Checkpoint not found at {weights_path}. Running base weights.")
        
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
        
        # Matches your exact padding and normalization schemas
        if model_type == "mnist":
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Pad(2), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081])
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
        probs = F.softmax(outputs, dim=1).squeeze(0)

    labels = LABELS_MAP[model_type]
    class_probs = {labels[i]: float(probs[i]) for i in range(10)}
    prediction = max(class_probs, key=class_probs.get)

    return {
        "active_model": model_type,
        "prediction": prediction,
        "confidence": class_probs[prediction],
        "probabilities": class_probs
    }
