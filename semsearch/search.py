"""KNN search over the FAISS index, with BM25 re-ranking via RRF."""

import json
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from semsearch.config import DATA_DIR, DEFAULT_K, EMBEDDING_MODEL, EMBEDDING_QUERY_PROMPT
from semsearch.models import SearchResult

_RRF_K = 60

# Normalized kind categories → sets of raw keywords from .v files
KIND_CATEGORIES: dict[str, set[str]] = {
    "theorem": {"Theorem", "Lemma", "Fact", "Remark", "Corollary", "Proposition", "Property", "Example"},
    "definition": {"Definition", "Fixpoint", "CoFixpoint"},
    "type": {"Inductive", "CoInductive", "Record", "Structure"},
    "class": {"Class", "Instance"},
    "axiom": {"Axiom", "Parameter", "Hypothesis", "Variable"},
}


def _make_text(decl: dict) -> str:
    annotation = decl.get("annotation")
    if annotation:
        return f"{decl['name']}: {annotation}"
    type_str = decl["type"]
    if len(type_str) > 500:
        type_str = type_str[:500] + "..."
    return f"{decl['name']}: {type_str}"


class SearchEngine:
    """Hybrid semantic + BM25 search over Coq declarations."""

    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            data_dir = DATA_DIR

        index_path = data_dir / "index.faiss"
        metadata_path = data_dir / "metadata.jsonl"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Index not found at {data_dir}. Run `semsearch-embed` first."
            )

        self.index = faiss.read_index(str(index_path))
        self.metadata: list[dict] = []
        with open(metadata_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Build BM25 index from the same texts used for embedding
        texts = [_make_text(m) for m in self.metadata]
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        library: str | None = None,
        kind: str | None = None,
    ) -> list[SearchResult]:
        """Search for declarations matching the query.

        Args:
            query: Natural language query
            k: Number of results to return
            library: Optional filter by library name
            kind: Optional filter by kind category: theorem, definition, type, class, axiom
        """
        filtering = bool(library or kind)
        candidates = k * 10 if filtering else k * 5
        kind_keywords = KIND_CATEGORIES.get(kind.lower(), {kind}) if kind else None

        # --- Vector search ---
        q_embedding = self.model.encode(
            [query], prompt=EMBEDDING_QUERY_PROMPT, normalize_embeddings=True
        )
        q_embedding = np.array(q_embedding, dtype=np.float32)
        vec_scores, vec_indices = self.index.search(
            q_embedding, min(candidates, self.index.ntotal)
        )
        vec_ranks = {
            int(idx): rank
            for rank, (idx, score) in enumerate(zip(vec_indices[0], vec_scores[0]))
            if idx >= 0
        }

        # --- BM25 search ---
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:candidates]
        bm25_ranks = {int(idx): rank for rank, idx in enumerate(bm25_top)}

        # --- RRF fusion ---
        all_indices = set(vec_ranks) | set(bm25_ranks)
        rrf_scores = {
            idx: (
                (1 / (_RRF_K + vec_ranks[idx]) if idx in vec_ranks else 0)
                + (1 / (_RRF_K + bm25_ranks[idx]) if idx in bm25_ranks else 0)
            )
            for idx in all_indices
        }
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked:
            meta = self.metadata[idx]
            if library and meta.get("library") != library:
                continue
            if kind_keywords and meta.get("kind") not in kind_keywords:
                continue
            results.append(
                SearchResult(
                    name=meta["name"],
                    type=meta["type"],
                    kind=meta.get("kind"),
                    library=meta["library"],
                    module=meta["module"],
                    annotation=meta.get("annotation"),
                    score=score,
                )
            )
            if len(results) >= k:
                break

        return results
