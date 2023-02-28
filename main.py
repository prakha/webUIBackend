from fastapi import FastAPI
from qdrant_client import QdrantClient
import json

app = FastAPI()

qdrant_client = QdrantClient(address='http://localhost:6333')

@app.post("/create_index/{index_name}")
def create_index(index_name: str, dimension: int):
    qdrant_client.create_index(index_name=index_name, dimension=dimension)
    return {"message": f"Index {index_name} created with {dimension} dimensions."}

@app.post("/upload_embeddings/{index_name}")
def upload_embeddings(index_name: str, embeddings_file: str):
    with open(embeddings_file, 'r') as f:
        embeddings = json.load(f)

    vectors = [
        {'id': idx, 'vector': emb} for idx, emb in embeddings.items()
    ]

    qdrant_client.upsert_entities(index_name=index_name, entities=vectors)
    return {"message": f"{len(vectors)} embeddings uploaded to index {index_name}."}

@app.get("/search_embeddings/{index_name}")
def search_embeddings(index_name: str, vector: list, top: int):
    search_results = qdrant_client.search(index_name=index_name, vector=vector, top=top)
    return {"search_results": search_results}
