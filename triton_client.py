import tritonclient.http as httpclient
from transformers import AutoTokenizer
import numpy as np
import os

LOCAL_MODEL_PATH = os.path.abspath("/home/trainee1/failed_root/raw_models/all-mpnet-base-v2/")
RERANKER_MODEL_PATH = os.path.abspath("/home/trainee1/failed_root/raw_models/bge-reranker-large/")

model_tokenizers = {
    "mpnet_model": AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH),
    "bge_reranker": AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
}

triton_client = httpclient.InferenceServerClient(url="localhost:8000")

def get_embedding(text, model_name="mpnet_model"):
    tokenizer = model_tokenizers[model_name]
    tokens = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
    input_ids = tokens["input_ids"].astype(np.int64)
    attention_mask = tokens["attention_mask"].astype(np.int64)

    inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)

    response = triton_client.infer(model_name=model_name, inputs=inputs)
    return response.as_numpy("output__0")[0]

def get_reranker_score(pairs: list, model_name="bge_reranker"):
    tokenizer = model_tokenizers[model_name]
    batch = tokenizer(pairs, padding=True, truncation=True, return_tensors="np", max_length=512)
    input_ids = batch["input_ids"].astype(np.int64)
    attention_mask = batch["attention_mask"].astype(np.int64)

    inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)

    response = triton_client.infer(model_name=model_name, inputs=inputs)
    logits = response.as_numpy("logits")
    if logits is None:
       raise RuntimeError("No logits received from Triton")

    return np.atleast_1d(logits).flatten()
