import os
import torch
import pandas as pd
from sentence_transformers import util
from triton_client import get_embedding, get_reranker_score

CACHE_PATH = "data/mpnet_label_embeddings.pt"
CSV_PATH = "data/label_descriptions.csv"
LOCAL_MODEL_PATH = os.path.abspath("/home/trainee1/failed_root/raw_models/all-mpnet-base-v2/")

from transformers import AutoTokenizer, AutoModel

def load_or_compute_label_embeddings():
    if os.path.exists(CACHE_PATH):
        return torch.load(CACHE_PATH, map_location="cuda")
    else:
        df = pd.read_csv(CSV_PATH)
        df = df.dropna(subset=["Description"])
        descriptions = df["Description"].tolist()

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to("cuda")

        with torch.no_grad():
            embeddings = []
            for desc in descriptions:
                tokens = tokenizer(desc, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
                output = model(**tokens).last_hidden_state.mean(dim=1)
                embeddings.append(output.squeeze(0))
            stacked = torch.stack(embeddings).to("cuda")
            normed = torch.nn.functional.normalize(stacked, dim=1)
        torch.save(normed, CACHE_PATH)
        return normed

def get_top_k_matches(text: str, top_k: int = 5, model_name: str = "mpnet_model"):
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Discipline", "Subject"])
    label_embeddings = load_or_compute_label_embeddings()

    text_embedding = torch.tensor(get_embedding(text, model_name=model_name), device="cuda")
    text_embedding = torch.nn.functional.normalize(text_embedding.unsqueeze(0), dim=1)

    cosine_scores = util.cos_sim(text_embedding, label_embeddings)[0]
    top_10_indices = torch.topk(cosine_scores, k=10).indices

    top_labels = [
        (
            df.iloc[int(idx)]["Discipline"],
            df.iloc[int(idx)]["Subject"],
            df.iloc[int(idx)]["Description"],
            float(cosine_scores[idx])
        )
        for idx in top_10_indices
    ]

    pairs = [(text, desc) for (_, _, desc, _) in top_labels if isinstance(desc, str)]

    scores = get_reranker_score(pairs)
    if isinstance(scores, float):
        scores = [scores]

    reranked = list(zip(top_labels, scores))
    reranked.sort(key=lambda x: x[1], reverse=True)

    return {
        "text": text,
        "matches": [
            {
                "label": f"{d.strip()} - {s.strip()}",
                "cosine_score": round(float(cos_score), 4),
                "rerank_score": round(float(score), 4)
            }
            for ((d, s, _, cos_score), score) in reranked[:top_k]
        ]
    }

def get_top_k_matches_batch(docs: list, top_k: int = 5, model_name: str = "mpnet_model"):
    results = []
    for doc in docs:
        doc_id = doc.get("id")
        text = doc.get("text")
        if not doc_id or not text:
            continue
        match = get_top_k_matches(text, top_k=top_k, model_name=model_name)
        results.append({"id": doc_id, "result": match})
    return {"results": results}

def generate_cache_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Description"])
    descriptions = df["Description"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to("cuda")

    with torch.no_grad():
        embeddings = []
        for desc in descriptions:
            tokens = tokenizer(desc, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
            output = model(**tokens).last_hidden_state.mean(dim=1)
            embeddings.append(output.squeeze(0))
        stacked = torch.stack(embeddings).to("cuda")
        normed = torch.nn.functional.normalize(stacked, dim=1)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    torch.save(normed, f"data/{base_name}_embeddings.pt")

    return normed
