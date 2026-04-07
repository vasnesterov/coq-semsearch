"""KNN search over the FAISS index."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from semsearch.config import DATA_DIR, DEFAULT_K, EMBEDDING_MODEL
from semsearch.models import SearchResult


class SearchEngine:
    """Semantic search over Coq declarations."""

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

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        library: str | None = None,
    ) -> list[SearchResult]:
        """Search for declarations matching the query.

        Args:
            query: Natural language query
            k: Number of results to return
            library: Optional filter by library name
        """
        # If filtering by library, we need to search more and then filter
        search_k = k * 5 if library else k

        # Embed query
        q_embedding = self.model.encode(
            [query], normalize_embeddings=True
        )
        q_embedding = np.array(q_embedding, dtype=np.float32)

        # Search
        scores, indices = self.index.search(q_embedding, min(search_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]

            if library and meta.get("library") != library:
                continue

            results.append(
                SearchResult(
                    name=meta["name"],
                    type=meta["type"],
                    kind=meta.get("kind"),
                    library=meta["library"],
                    module=meta["module"],
                    annotation=meta.get("annotation"),
                    score=float(score),
                )
            )

            if len(results) >= k:
                break

        return results
