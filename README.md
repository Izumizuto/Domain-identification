# Domain-identification
Domain Identification in Work Reports using Transformers (MPNet + BGE Reranker, Triton, FastAPI)
# Domain Identification in Work Reports Using Transformers

## 📌 Overview
This project performs **automatic domain identification** in technical work reports using a **hybrid NLP architecture**:
- **Bi-encoder (all-mpnet-base-v2)** for fast semantic retrieval
- **Cross-encoder (bge-reranker-large)** for precision reranking
- Deployed on **NVIDIA Triton Inference Server** for scalable inference
- Served via **FastAPI** for easy API access

---

## 🚀 Architecture
1. **Stage 1: Efficient Retrieval**
   - Precomputed label embeddings stored as PyTorch tensors
   - Cosine similarity search for top-k candidate labels

2. **Stage 2: Precision Reranking**
   - Cross-encoder re-evaluates candidate matches
   - Produces final sorted domain predictions

---

## 📂 Project Structure
Domain-identification/
│── README.md
│── LICENSE
│── requirements.txt
│── main.py
│── triton_client.py
│── logic/
│ └── match_labels.py
│── data/
│ └── label_descriptions.csv
│── model-repository/
│ ├── mpnet_model/config.pbtxt
│ └── bge_reranker/config.pbtxt
│── docs/
│ └── project_presentation.pdf
│── .gitignore

yaml
Copy
Edit

---

## ⚠️ Large Files Excluded from GitHub
The following are **NOT included in this repository** due to size limitations:
- `*.pt` model weights  
- HuggingFace cache (`raw_models/`)  
- Generated embedding caches (`*_embeddings.pt`)  

---

## 📥 Model Setup
Before running the project, you must download and place the models locally.

### **1️⃣ Download MPNet Model**
```bash
git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
Place inside:

bash
Copy
Edit
raw_models/all-mpnet-base-v2/
2️⃣ Download BGE Reranker Model
bash
Copy
Edit
git clone https://huggingface.co/BAAI/bge-reranker-large
Place inside:

bash
Copy
Edit
raw_models/bge-reranker-large/
3️⃣ Export Models for Triton
Convert the models to TorchScript or ONNX and place in:

bash
Copy
Edit
model-repository/mpnet_model/1/model.pt
model-repository/bge_reranker/1/model.pt
⚙️ Setup & Run
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Start Triton Inference Server

bash
Copy
Edit
docker run --gpus=all --rm -p8000:8000 -p8001:8001 \
   -v $(pwd)/model-repository:/models nvcr.io/nvidia/tritonserver:23.08-py3 \
   tritonserver --model-repository=/models
Start FastAPI

bash
Copy
Edit
uvicorn main:app --reload --host 0.0.0.0 --port 8080
🔌 API Endpoints
1. Embed
bash
Copy
Edit
POST /embed
Form:

text or file (JSON batch)

pass_key

top_k (optional)

2. Generate Cache
bash
Copy
Edit
POST /generate-cache
Form:

file (CSV)

pass_key

📌 Future Work
Model versioning for easy upgrades

Add Wikipedia API for label expansion

Optimize batch inference in production
