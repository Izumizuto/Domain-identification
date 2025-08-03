# Domain Identification in Work Reports Using Transformers

## Overview
This project performs **automatic domain identification** in technical work reports using a hybrid NLP architecture:
- **Bi-encoder** (`all-mpnet-base-v2`) for fast semantic retrieval  
- **Cross-encoder** (`bge-reranker-large`) for precision reranking  
- Deployed on **NVIDIA Triton Inference Server** for scalable inference  
- Served via **FastAPI** for easy API access  

---

## Architecture
### Stage 1: Efficient Retrieval
- Precomputed label embeddings stored as PyTorch tensors  
- Cosine similarity search to retrieve top-k candidate labels  

### Stage 2: Precision Reranking
- Cross-encoder re-evaluates candidate matches  
- Produces final sorted domain predictions  

---

## Project Structure
```
Domain-identification/
│── README.md
│── LICENSE
│── requirements.txt
│── main.py
│── triton_client.py
│── logic/
│   └── match_labels.py
│── data/
│   └── label_descriptions.csv
│── model-repository/
│   ├── mpnet_model/config.pbtxt
│   └── bge_reranker/config.pbtxt
│── docs/
│   └── project_presentation.pdf
│── .gitignore
```

---

## Large Files Excluded from GitHub
The following are **not included** in the repository due to size limitations:
- `*.pt` model weights  
- HuggingFace cache (`raw_models/`)  
- Generated embedding caches (`*_embeddings.pt`)  

---

## Model Setup
Before running the project, download and place the models locally.

### 1. Download MPNet Model
```bash
git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
```
Place inside:
```
raw_models/all-mpnet-base-v2/
```

### 2. Download BGE Reranker Model
```bash
git clone https://huggingface.co/BAAI/bge-reranker-large
```

### 3. Export Models for Triton
Convert models to TorchScript or ONNX and place them in:
```
model-repository/mpnet_model/1/model.pt
model-repository/bge_reranker/1/model.pt
```

---

## Setup & Run
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Triton Inference Server
```bash
docker run --gpus=all --rm -p8000:8000 -p8001:8001 \
   -v $(pwd)/model-repository:/models nvcr.io/nvidia/tritonserver:23.08-py3 \
   tritonserver --model-repository=/models
```

### 3. Start FastAPI
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

---

## Future Work
- Model versioning for easy upgrades  
- Add Wikipedia API for label expansion  
- Optimize batch inference in production  
