from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Optional
from logic.match_labels import get_top_k_matches, get_top_k_matches_batch
import json
import shutil
from logic.match_labels import generate_cache_from_csv

app = FastAPI()
PASS_KEY = "manas123"  # Change this securely

@app.post("/embed")
def embed_single(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    model_name: Optional[str] = Form("mpnet_model"),
    top_k: Optional[int] = Form(5),
    pass_key: str = Form(...)
):
    if pass_key != PASS_KEY:
        raise HTTPException(status_code=403, detail="Invalid pass_key")

    if not text and not file:
        raise HTTPException(status_code=400, detail="Either text or file must be provided")

    if file:
        contents = file.file.read()
        try:
            batch_json = json.loads(contents)
            if not isinstance(batch_json, list):
                raise ValueError
            return get_top_k_matches_batch(batch_json, top_k=top_k, model_name=model_name)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON file uploaded")

    return get_top_k_matches(text, top_k=top_k, model_name=model_name)
   
@app.post("/generate-cache")
def generate_cache(file: UploadFile = File(...), pass_key: str = Form(...)):
    if pass_key != PASS_KEY:
        raise HTTPException(status_code=403, detail="Invalid pass_key")

    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        generate_cache_from_csv(file_location)
        return {"status": "success", "message": f"Cache generated from {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
