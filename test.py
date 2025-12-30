import json
import os
from typing import List, Dict
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

from collections import defaultdict

from sentence_transformers import SentenceTransformer
import re
from bs4 import BeautifulSoup
from bs4.element import Tag
from FlagEmbedding import FlagReranker

with open('image_extracted_gpu.json', 'r', encoding = 'utf-8') as json_file:
    pages = json.load(json_file)



def sanitize_metadata(meta: dict):
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""        # safest default
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)    # fallback (shouldn't happen)
    return clean

def flatten_page_blocks(blocks):
    """
    Returns:
    - embed_text: string used for embedding
    - html_tables: list of raw HTML tables (metadata only)
    """

    lines = []
    html_tables = []

    for block in blocks:
        btype = block.get("type")
        content = block.get("content")

        if not content:
            continue

        if btype == "page_number":
            lines.append(f"PAGE {content}")

        elif btype == "title":
            lines.append(f"TITLE: {content}")

        elif btype == "header":
            lines.append(f"HEADER: {content}")

        elif btype == "text":
            lines.append(content)

        elif btype == "list":
            lines.append(f"LIST: {content}")

        elif btype == "table_caption":
            lines.append(f"TABLE CAPTION: {content}")

        elif btype == "table":
            html_tables.append(content)

            soup = BeautifulSoup(content, "html.parser")
            for row in soup.find_all("tr"):
                cells = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
                if cells:
                    lines.append(" | ".join(cells))

    return "\n".join(lines), html_tables

page_docs = []
page_metas = []

for page in pages:
    blocks = page.get("blocks", [])
    image_name = page.get("image")

    embed_text, html_tables = flatten_page_blocks(blocks)

    if not embed_text.strip():
        continue

    page_no = next(
        (b["content"] for b in blocks if b.get("type") == "page_number"),
        ""
    )

    page_docs.append(embed_text)

    meta = {
        "page": page_no or "",
        "image": image_name or "",
        "type": "page",
        "html_tables": "\n\n".join(html_tables) if html_tables else ""
    }

    page_metas.append(sanitize_metadata(meta))

embedder = SentenceTransformer("./bge-large-en")


client = chromadb.Client(
    Settings(
        persist_directory="./chroma_mineru_page_wise",
        anonymized_telemetry=False
    )
)

page_embeddings = embedder.encode(
    page_docs,                     
    show_progress_bar=True,
    normalize_embeddings=True
)

page_col = client.create_collection("page_index")
# page_col = client.get_collection("page_index")

page_col.add(
    documents=page_docs,
    embeddings=page_embeddings,
    ids=[f"text_{i}" for i in range(len(page_docs))],
    metadatas=page_metas           
)

def embed_query(query: str):
    return embedder.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True
    )

query = "ClearDiagnosticInformation "
q_emb = embed_query(query)

page_hits = page_col.query(
    query_embeddings=[q_emb],
    n_results=20,
    include=["documents", "metadatas", "distances"]
)
def flatten_hits(hits, source_type):
    docs = []
    for doc_id, doc, meta, dist in zip(
        hits["ids"][0],          # ← IDs are HERE
        hits["documents"][0],
        hits["metadatas"][0],
        hits["distances"][0],
    ):
        docs.append({
            "id": doc_id,
            "content": doc,
            "metadata": meta,
            "distance": dist,
            "source": source_type
        })
    return docs

table_docs = flatten_hits(page_hits, "table")
reranker = FlagReranker(
    "./bge-reranker-v2-m3",
    use_fp16=True  
)

def rerank_results(query, retrieved_docs, reranker, top_k=5):
    pairs = [[query, d["content"]] for d in retrieved_docs]
    scores = reranker.compute_score(pairs, normalize=True)

    reranked = sorted(
        zip(retrieved_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return reranked[:top_k]

final_results = rerank_results(
    query,
    table_docs,
    reranker,
    top_k=5
)

top_k = 5
final_results = final_results[:top_k]

for doc, score in final_results:
    print(
        f"score={score:.4f} | source={doc['source']} | page={doc['metadata'].get('page')}"
    )















