#!/usr/bin/env python3
"""
extract.py — Terminal RAG chatbot over the LabDocs index using Ollama

Build index first with ingest.py, then:

  python3 extract.py \
    --index ./idx \
    --model qwen2.5:3b-instruct \
    --embed-model nomic-embed-text \
    --topk 8

Commands in chat:
  /help     - show help
  /sources  - show the chosen source chunks for last answer
  /raw      - toggle showing raw context before answers
  /exit     - quit
"""

import argparse, json, os, re, sys, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# --- deps -------------------------------------------------------------------
try:
    import numpy as np
except Exception:
    print("[error] numpy required: pip install numpy", file=sys.stderr); raise

try:
    import ollama
except Exception:
    print("[error] ollama python client required: pip install ollama", file=sys.stderr); raise

# BM25 (pure python, light)
try:
    from rank_bm25 import BM25Okapi
except Exception:
    print("[error] rank-bm25 required: pip install rank-bm25", file=sys.stderr); raise

# --- IO ---------------------------------------------------------------------
def load_index(index_dir: Path) -> Tuple["np.ndarray", List[Dict]]:
    emb = np.load(index_dir / "embeddings.npy", mmap_mode="r")
    metas: List[Dict] = []
    with open(index_dir / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    if emb.shape[0] != len(metas):
        raise RuntimeError("embeddings count != meta lines")
    return emb, metas

# --- tokenization -----------------------------------------------------------
STOP = set("""
a an the of and or for to in with from on by as is are was were be been being
this that those these into using via use used based per at it its their his her
we you they he she i our your not null none but if then than over under
""".split())

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")  # simple alnum words

def tokenize(s: str) -> List[str]:
    return [w.lower() for w in TOKEN_RE.findall(s) if w.lower() not in STOP and len(w) > 1]

# --- retrieval --------------------------------------------------------------
def build_bm25(corpus_texts: List[str]) -> BM25Okapi:
    tokenized = [tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokenized), tokenized

def bm25_scores(bm25: BM25Okapi, tokenized_corpus: List[List[str]], q: str) -> "np.ndarray":
    tq = tokenize(q)
    sc = bm25.get_scores(tq)  # numpy array
    return sc

def embed_query(q: str, embed_model: str) -> "np.ndarray":
    resp = ollama.embeddings(model=embed_model, prompt=q)
    v = np.array(resp["embedding"], dtype="float32")
    v /= (np.linalg.norm(v) + 1e-9)
    return v

def topk_indices(scores: "np.ndarray", k: int) -> List[int]:
    k = min(k, scores.shape[0])
    if k <= 0:
        return []
    # argpartition then sort
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.tolist()

# --- context packing --------------------------------------------------------
def format_span(m: Dict, idx: int, max_words: int = 180) -> Tuple[str, str]:
    words = m["text"].split()
    snippet = " ".join(words[:max_words])
    label   = f"[{idx}] (doc={Path(m['doc']).name}, page={m['page']})"
    body    = f"{label}\n{snippet}"
    return body, label

def build_context(metas: List[Dict], chosen_ids: List[int], max_words_per_span=180) -> Tuple[str, List[str]]:
    blocks, labels = [], []
    for i, cid in enumerate(chosen_ids, 1):
        body, label = format_span(metas[cid], i, max_words_per_span)
        labels.append(label)
        blocks.append(body)
    ctx = "Context:\n" + "\n\n".join(blocks)
    return ctx, labels

# --- scoring fusion ---------------------------------------------------------
def fused_topk(
    query: str,
    emb_mat: "np.ndarray",
    metas: List[Dict],
    bm25: BM25Okapi,
    tokenized_corpus: List[List[str]],
    embed_model: str,
    k_vector: int = 100,
    k_bm25: int = 100,
    k_final: int = 8,
    alpha: float = 0.65,   # weight for vector score
) -> Tuple[List[int], Dict[str, float]]:
    qv = embed_query(query, embed_model)  # [D]
    # cosine since both sides L2-normalized
    vec_scores = emb_mat @ qv  # [N]

    # lexical
    bm_scores  = bm25_scores(bm25, tokenized_corpus, query)  # [N]

    # restrict to candidates (union of top vector + top bm25)
    topv = set(topk_indices(vec_scores, k_vector))
    topb = set(topk_indices(bm_scores,  k_bm25))
    cand = list(topv.union(topb))
    if not cand:
        return [], {}

    # normalize scores over candidates
    vs = vec_scores[cand]
    bs = bm_scores[cand]
    vs_n = (vs - vs.min()) / (vs.max() - vs.min() + 1e-9)
    bs_n = (bs - bs.min()) / (bs.max() - bs.min() + 1e-9)

    fused = alpha * vs_n + (1 - alpha) * bs_n
    order = np.argsort(fused)[::-1]
    chosen = [cand[i] for i in order[:k_final]]

    dbg = {
        "cand": len(cand),
        "alpha": alpha,
        "best_vec": float(vs.max()),
        "best_bm25": float(bs.max())
    }
    return chosen, dbg

# --- chat loop --------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an assistant for laboratory documentation. "
    "Answer STRICTLY from the provided Context. "
    "If the answer is not present, say 'Not found in the provided documents.' "
    "After the answer, include a short 'Where this comes from' list using the labels [1],[2], etc."
)

def chat_loop(
    emb_mat: "np.ndarray",
    metas: List[Dict],
    model: str,
    embed_model: str,
    topk: int,
    raw_toggle: bool
):
    print(f"[ready] model={model} embed_model={embed_model} chunks={emb_mat.shape[0]}")
    # Prepare BM25 once
    corpus_texts = [m["text"] for m in metas]
    bm25, tokenized_corpus = build_bm25(corpus_texts)

    last_labels: List[str] = []

    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not q:
            continue
        if q in {"/exit", ":q", "quit"}:
            break
        if q == "/help":
            print("Commands: /sources, /raw (toggle), /exit")
            continue
        if q == "/raw":
            raw_toggle = not raw_toggle
            print(f"[toggle] raw context: {'ON' if raw_toggle else 'OFF'}")
            continue
        if q == "/sources":
            if not last_labels:
                print("[info] no previous answer yet.")
            else:
                print("\n".join(last_labels))
            continue

        t0 = time.time()
        chosen_ids, dbg = fused_topk(
            q, emb_mat, metas, bm25, tokenized_corpus, embed_model,
            k_vector=100, k_bm25=100, k_final=topk, alpha=0.65
        )
        if not chosen_ids:
            print("No relevant chunks found.")
            continue

        ctx, labels = build_context(metas, chosen_ids, max_words_per_span=180)
        last_labels = labels

        if raw_toggle:
            print("\n--- RAW CONTEXT ---")
            print(ctx)
            print("--- END CONTEXT ---\n")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {q}\n\n{ctx}\n\nOutput format:\n"
                                        "- 2–5 sentence answer\n"
                                        "- Then 'Where this comes from' with [1], [2] ..."}
        ]

        # Stream the answer for a snappier UX
        print("\n", end="", flush=True)
        try:
            stream = ollama.chat(model=model, messages=messages, stream=True)
            for part in stream:
                chunk = part.get("message", {}).get("content", "")
                if chunk:
                    print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n[error] ollama.chat failed: {e}", file=sys.stderr)
            continue

        dt = time.time() - t0
        print(f"\n\n[info] retrieved {len(chosen_ids)} spans in {dt:.2f}s")
        print("[sources]")
        for i, cid in enumerate(chosen_ids, 1):
            m = metas[cid]
            print(f"  [{i}] {Path(m['doc']).name}  page={m['page']}  path={m['doc']}")

# --- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to index dir produced by ingest.py")
    ap.add_argument("--model", default="qwen2.5:3b-instruct", help="Ollama chat model")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model (must match ingest)")
    ap.add_argument("--topk", type=int, default=8, help="number of context chunks")
    ap.add_argument("--raw", action="store_true", help="print raw context before answering")
    args = ap.parse_args()

    index_dir = Path(args.index).resolve()
    try:
        _ = ollama.list()
    except Exception as e:
        print("[connection error] Could not reach Ollama. Start it with `ollama serve`.", file=sys.stderr)
        raise

    emb, metas = load_index(index_dir)
    # sanity: ensure embedding dims match (we don't enforce, but warn if ingest used a different model)
    vm_file = index_dir / "vector_model.txt"
    if vm_file.exists():
        ingested_model = vm_file.read_text(encoding="utf-8").strip()
        if ingested_model != args.embed_model:
            print(f"[warn] ingest used embed model '{ingested_model}', but you're using '{args.embed_model}'. "
                  f"Cosine still works if dims match.", file=sys.stderr)

    chat_loop(emb, metas, args.model, args.embed_model, args.topk, args.raw)

if __name__ == "__main__":
    main()
