

import os
from typing import List, Dict
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer






embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts):
    return embedder.encode(texts, show_progress_bar=False)



chroma_client = chromadb.Client(
    Settings(
        persist_directory="./chroma_rnd",
        anonymized_telemetry=False
    )
)





