#!/usr/bin/env python3
"""
ingest.py — Build a lightweight retrieval index from LabDocs

Usage:
  python3 ingest.py \
    --data ~/Documents/human/adlm/LabDocs \
    --index ./idx \
    --embed-model nomic-embed-text \
    --max-words 800 --overlap 120

Outputs (in --index dir):
  - embeddings.npy       (float32, L2-normalized, shape [N, D])
  - meta.jsonl           (one JSON per chunk: {id, doc, page, ext, text})
  - stats.json           (counts, dim, elapsed)
  - vector_model.txt     (just the name you used)
"""

import argparse, json, os, re, sys, time, math
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Optional

# --- deps (minimal) ---------------------------------------------------------
try:
    import numpy as np
except Exception as e:
    print("[error] numpy required: pip install numpy", file=sys.stderr); raise

try:
    import ollama
except Exception as e:
    print("[error] ollama python client required: pip install ollama", file=sys.stderr); raise

# PDF extractors: prefer PyMuPDF; fallback to PyPDF2 basic text
def _try_import_pymupdf():
    try:
        import fitz  # PyMuPDF
        return fitz
    except Exception:
        return None

def _try_import_pypdf():
    try:
        from PyPDF2 import PdfReader
        return PdfReader
    except Exception:
        return None

# --- text utils -------------------------------------------------------------
WHITESPACE_RE = re.compile(r"[ \t]+")
MULTINL_RE    = re.compile(r"\n{3,}")

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = WHITESPACE_RE.sub(" ", s)
    s = s.replace("\r", "\n")
    s = MULTINL_RE.sub("\n\n", s).strip()
    return s

def split_words(text: str, max_words=800, overlap=120) -> Iterator[Tuple[int, int, str]]:
    """Yield (start_idx, end_idx, chunk_text) by word windows."""
    words = text.split()
    if not words:
        return
    n = len(words)
    step = max(1, max_words - overlap)
    for start in range(0, n, step):
        end = min(n, start + max_words)
        yield start, end, " ".join(words[start:end])
        if end >= n:
            break

# --- extractors -------------------------------------------------------------
def extract_pdf_text(path: Path) -> List[Tuple[int, str]]:
    """Return list of (page_number_1based, text)."""
    fitz = _try_import_pymupdf()
    if fitz is not None:
        doc = fitz.open(str(path))
        out = []
        for i, page in enumerate(doc, 1):
            t = page.get_text("text") or ""
            t = clean_text(t)
            if t:
                out.append((i, t))
        doc.close()
        return out

    PdfReader = _try_import_pypdf()
    if PdfReader is not None:
        try:
            rd = PdfReader(str(path))
            out = []
            for i, p in enumerate(rd.pages, 1):
                t = p.extract_text() or ""
                t = clean_text(t)
                if t:
                    out.append((i, t))
            return out
        except Exception:
            return []

    print(f"[warn] No PDF extractor available for {path}. Install PyMuPDF or PyPDF2.", file=sys.stderr)
    return []

def extract_text_file(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return clean_text(raw)
    except Exception:
        return ""

# --- embedding --------------------------------------------------------------
def embed_one(text: str, model: str) -> List[float]:
    # Ollama embeddings: ollama.embeddings(model=..., prompt=...)
    resp = ollama.embeddings(model=model, prompt=text)
    emb = resp.get("embedding")
    if not emb:
        raise RuntimeError("No embedding returned by Ollama")
    return emb

def l2_normalize_inplace(mat: "np.ndarray") -> None:
    # mat shape: [N, D]
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat /= norms

# --- crawl ------------------------------------------------------------------
def find_files(root: Path) -> Iterator[Path]:
    exts = {".pdf", ".txt", ".md", ".markdown"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

# --- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to LabDocs folder")
    ap.add_argument("--index", required=True, help="Output index dir")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    ap.add_argument("--max-words", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--limit-files", type=int, default=0, help="For testing; 0 = all")
    args = ap.parse_args()

    data_dir  = Path(os.path.expanduser(args.data)).resolve()
    index_dir = Path(args.index).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    # quick health check: list models so we fail fast if ollama isn't up
    try:
        _ = ollama.list()
    except Exception as e:
        print("[connection error] Could not reach Ollama. Is it running? (e.g., `ollama serve`)", file=sys.stderr)
        raise

    files = list(find_files(data_dir))
    files.sort()
    if args.limit_files > 0:
        files = files[: args.limit_files]

    print(f"[ingest] scanning: {data_dir}")
    print(f"[ingest] files: {len(files)}")
    print(f"[ingest] embedding model: {args.embed_model}")

    meta_path = index_dir / "meta.jsonl"
    emb_path  = index_dir / "embeddings.npy"
    stats_path= index_dir / "stats.json"
    (index_dir / "vector_model.txt").write_text(args.embed_model, encoding="utf-8")

    # Collect in memory (simple, good enough for MVP)
    embeddings: List[List[float]] = []
    metas: List[Dict] = []

    t0 = time.time()
    chunk_id = 0
    for fi, f in enumerate(files, 1):
        ext = f.suffix.lower()
        try:
            if ext == ".pdf":
                pages = extract_pdf_text(f)  # list[(page, text)]
                for page_num, page_text in pages:
                    for _, _, chunk in split_words(page_text, args.max_words, args.overlap):
                        if not chunk.strip():
                            continue
                        emb = embed_one(chunk, args.embed_model)
                        embeddings.append(emb)
                        metas.append({
                            "id": chunk_id,
                            "doc": str(f),
                            "page": page_num,
                            "ext": "pdf",
                            "text": chunk
                        })
                        chunk_id += 1
            else:
                txt = extract_text_file(f)
                if not txt:
                    continue
                for _, _, chunk in split_words(txt, args.max_words, args.overlap):
                    if not chunk.strip():
                        continue
                    emb = embed_one(chunk, args.embed_model)
                    embeddings.append(emb)
                    metas.append({
                        "id": chunk_id,
                        "doc": str(f),
                        "page": None,
                        "ext": ext.lstrip("."),
                        "text": chunk
                    })
                    chunk_id += 1
        except KeyboardInterrupt:
            print("\n[ingest] interrupted by user", file=sys.stderr)
            break
        except Exception as e:
            print(f"[warn] skipping {f} due to error: {e}", file=sys.stderr)
            continue

        if fi % 25 == 0:
            elapsed = time.time() - t0
            print(f"[ingest] processed {fi}/{len(files)} files | chunks={chunk_id} | {elapsed:.1f}s")

    if not embeddings:
        print("[error] no chunks embedded. Nothing to write.", file=sys.stderr)
        sys.exit(1)

    # Write meta.jsonl
    with open(meta_path, "w", encoding="utf-8") as fo:
        for m in metas:
            fo.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Write embeddings.npy (float32 normalized)
    emb_arr = np.array(embeddings, dtype="float32")
    l2_normalize_inplace(emb_arr)
    np.save(emb_path, emb_arr)

    elapsed = time.time() - t0
    stats = {
        "files": len(files),
        "chunks": int(emb_arr.shape[0]),
        "dim": int(emb_arr.shape[1]),
        "elapsed_sec": round(elapsed, 2)
    }
    with open(stats_path, "w", encoding="utf-8") as fo:
        json.dump(stats, fo, indent=2)
    print(f"[done] wrote index: {index_dir} | chunks={stats['chunks']} dim={stats['dim']} time={stats['elapsed_sec']}s")

if __name__ == "__main__":
    main()
