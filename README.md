Readme.md
1.	# PyTorch Production ML Pipeline Life Cycle
```
[ Local / CI Layer ]
Git Workflow -> GitHub Actions (PyTest)
│
▼
[ Containerization Layer ]
Docker Build Multi-Stage (Train & Serve Images)
│
▼
[ Kubernetes Orchestration ]
┌──────────────────────────────┴──────────────────────────────┐
▼ ▼
[ Training Stage ] [ Serving Stage ]
K8s Job Engine FastAPI Engine
│ │
▼ ▼
PersistentVolumeClaim ───► [ Persistent Storage ] ◄─── Mount Model Artifacts
```
2.	Python 3.11 standard environment structure.
3.	Multi-staged distinct Docker environments for train and serve,
4.	Kubernetes training tasks via declarative batch `Jobs`, outputting structured weight states to persistent cluster infrastructure, and serving via scaled horizontal `Deployments` with health validation endpoints (`/health`).
5.	Training epochs.
5.1.	 Cifar 10
```
6.	{"epoch": 16, "train_loss": 0.1754, "train_accuracy": 0.9379, "val_loss": 0.3234, "val_accuracy": 0.8988}
7.	{"event": "checkpoint_saved", "path": "D:\\Project\\best_custom_resnet_model.pth"}
8.	{"epoch": 17, "train_loss": 0.1594, "train_accuracy": 0.9434, "val_loss": 0.3599, "val_accuracy": 0.8949}
9.	{"epoch": 18, "train_loss": 0.1495, "train_accuracy": 0.9471, "val_loss": 0.3314, "val_accuracy": 0.899}
10.	{"epoch": 19, "train_loss": 0.1405, "train_accuracy": 0.9503, "val_loss": 0.3776, "val_accuracy": 0.8956}
11.	{"epoch": 20, "train_loss": 0.1309, "train_accuracy": 0.9542, "val_loss": 0.3309, "val_accuracy": 0.9056}
12.	{"epoch": 21, "train_loss": 0.1188, "train_accuracy": 0.9579, "val_loss": 0.3443, "val_accuracy": 0.9057}
13.	{"event": "early_stopping", "epoch": 21}
14.	{"event": "training_complete", "best_val_loss": 0.3234}
```
Fashion Mnist
```
{"event": "checkpoint_saved", "path": "D:\\Project\\checkpoints\\best_custom_mnist_model.pth"}
{"epoch": 7, "train_loss": 0.11, "train_accuracy": 0.9599, "val_loss": 0.231, "val_accuracy": 0.9268}
{"epoch": 8, "train_loss": 0.0939, "train_accuracy": 0.9658, "val_loss": 0.2661, "val_accuracy": 0.9149}
{"epoch": 9, "train_loss": 0.0742, "train_accuracy": 0.9728, "val_loss": 0.2198, "val_accuracy": 0.9349}
{"epoch": 10, "train_loss": 0.0613, "train_accuracy": 0.9781, "val_loss": 0.2558, "val_accuracy": 0.9329}
{"epoch": 11, "train_loss": 0.048, "train_accuracy": 0.9832, "val_loss": 0.3805, "val_accuracy": 0.9059}
{"event": "early_stopping", "epoch": 11}
{"event": "training_complete", "best_val_loss": 0.2172}
```
Directory Structure.
```
D:.
│   .gitignore
│
├───.github
│   └───workflows
│           ci.yml
│
├───checkpoints
│       best_custom_cifar10_model.pth
│       best_custom_mnist_model.pth
│
├───configs
│       training_config.yaml
│
├───data
├───docker
│       Dockerfile.serve
│       Dockerfile.train
│
├───k8s
│       configMap.yaml
│       hpa.yaml
│       namespace.yaml
│       pvc.yaml
│       serving_deployment.yaml
│       serving_service.yaml
│       training_job.yaml
│
├───requirements
│       serve.txt
│       train.txt
│
├───src
│       dataset.py
│       model.py
│       serve.py
│       train.py
│
└───tests
        test_cifar10.py
        test_fashion_mnist.py
```
##  Execution Lifecycles
### Local environment setup 
export CONFIG_PATH="configs/training_config.yaml" 
export DATA_DIR="./data" python src/train.py

### Spin up local API interfaces
export MODEL_PATH="/mnt/data/checkpoints/model.pth" 
uvicorn src:serve:app --host 0.0.0.0 --port 8000 –reload

### Compile the training runtime layer 
docker build -f docker/Dockerfile.train -t mlops- train:v1 .

docker build -f docker/Dockerfile.serve -t mlops- serve:v1 .

## Kubernetes Orchestration
### 1. Provision isolated structural logic spaces 
kubectl apply -f k8s/namespace.yaml 

### 2. Inject environment property parameter bindings 
kubectl apply -f k8s/configmap.yaml 

### 3. Provision persistent volumes to securely back up deep learning artifacts.  Ensure host environment provisions storage classes natively (such as minikube/kind default drivers) 

### 4. Trigger training orchestration jobs 
kubectl apply -f k8s/training-job.yaml 

### Track job progress and log extraction targets via standard streams: 
kubectl logs -n ml-training -l job-name= training-job --follow 

### 5. Boot high-availability API deployment vectors once training completes 
kubectl apply -f k8s/serving_deployment.yaml 
kubectl apply -f k8s/serving_service.yaml 

### 6. Apply dynamic elasticity properties 
kubectl apply -f k8s/hpa.yaml

###7. Verify pods are running and healthy:
kubectl get pods -n ml-training 
kubectl describe deployment model-serving -n ml-training


### Live Interface Health Status Verification 
curl -X GET http://localhost:8000/healthz ``` 
```json { "status": "healthy" } ```


### Send a prediction request
curl -X POST http://localhost:8080/predict \ -F "file=@D:\Project\data\mnist_images\sample_4212_Pullover.png

json { "active_model": fashion_mnist, "prediction": Pullover, "confidence": 99.81%}



json { "active_model": fashion_mnist, "prediction": Pullover, "confidence": 99.81%}



