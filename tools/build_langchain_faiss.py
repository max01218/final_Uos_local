#!/usr/bin/env python3
"""
Minimal script to build a LangChain-compatible FAISS index at ./embeddings/

Outputs:
  - embeddings/index.faiss
  - embeddings/index.pkl

It collects texts from:
  - icd11_ch6_data/raw/*.json (if present)
  - prompts/*.txt (optional, if present)

Embeddings model: sentence-transformers/all-mpnet-base-v2

Usage (from repo root):
  python -X utf8 tools/build_langchain_faiss.py \
    --input_raw_dir icd11_ch6_data/raw \
    --prompts_dir prompts \
    --out_dir embeddings \
    --model sentence-transformers/all-mpnet-base-v2 \
    --max_docs 5000 --chunk_size 1000 --chunk_overlap 100
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


def iter_json_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return directory.glob("*.json")


def read_text_files(directory: Path) -> List[Tuple[str, str]]:
    texts: List[Tuple[str, str]] = []
    if not directory.exists():
        return texts
    for p in directory.glob("*.txt"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if txt.strip():
                texts.append((txt, str(p)))
        except Exception:
            continue
    return texts


def flatten_json_to_text(obj: object, max_chars: int = 8000) -> str:
    parts: List[str] = []
    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
        elif isinstance(o, (str, int, float, bool)):
            parts.append(str(o))
        else:
            # ignore other types
            pass
    _walk(obj)
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks: List[str] = []
    if not text:
        return chunks
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def collect_corpus(raw_dir: Path, prompts_dir: Path, max_docs: int, chunk_size: int, chunk_overlap: int) -> Tuple[List[str], List[dict]]:
    texts: List[str] = []
    metas: List[dict] = []

    # From JSON raw files
    count = 0
    for jf in iter_json_files(raw_dir):
        if count >= max_docs:
            break
        try:
            with jf.open("r", encoding="utf-8", errors="ignore") as f:
                obj = json.load(f)
            full_text = flatten_json_to_text(obj)
            for chunk in chunk_text(full_text, chunk_size, chunk_overlap):
                texts.append(chunk)
                metas.append({"source": str(jf), "type": "icd11_raw"})
            count += 1
        except Exception:
            continue

    # From prompts (optional)
    for txt, path_str in read_text_files(prompts_dir):
        for chunk in chunk_text(txt, chunk_size, chunk_overlap):
            texts.append(chunk)
            metas.append({"source": path_str, "type": "prompt"})

    return texts, metas


def build_and_save_faiss(texts: List[str], metas: List[dict], model_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metas)
    vs.save_local(str(out_dir))


def main():
    parser = argparse.ArgumentParser(description="Build LangChain FAISS index into ./embeddings")
    parser.add_argument("--input_raw_dir", type=str, default="icd11_ch6_data/raw", help="Directory with ICD-11 raw JSON files")
    parser.add_argument("--prompts_dir", type=str, default="prompts", help="Directory with .txt prompts (optional)")
    parser.add_argument("--out_dir", type=str, default="embeddings", help="Output directory for FAISS index")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Sentence-Transformer model name")
    parser.add_argument("--max_docs", type=int, default=5000, help="Max number of JSON files to process")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for splitting text")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Overlap between chunks")
    args = parser.parse_args()

    raw_dir = Path(args.input_raw_dir)
    prompts_dir = Path(args.prompts_dir)
    out_dir = Path(args.out_dir)

    texts, metas = collect_corpus(raw_dir, prompts_dir, args.max_docs, args.chunk_size, args.chunk_overlap)
    if not texts:
        print("No texts collected. Please check input directories.")
        return

    print(f"Collected {len(texts)} chunks from corpus. Building embeddings with {args.model}...")
    build_and_save_faiss(texts, metas, args.model, out_dir)
    print(f"Saved FAISS store to: {out_dir}/index.faiss and {out_dir}/index.pkl")


if __name__ == "__main__":
    main()


