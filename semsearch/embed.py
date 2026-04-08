"""Build FAISS index from declarations (with or without annotations)."""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from rich.console import Console
from rich.progress import Progress
from sentence_transformers import SentenceTransformer

from semsearch.config import DATA_DIR, EMBEDDING_DIM, EMBEDDING_DOC_PROMPT, EMBEDDING_MODEL, EMBEDDING_QUERY_PROMPT

console = Console()


def load_declarations(data_dir: Path) -> list[dict]:
    """Load all declarations from all library subdirectories."""
    decls = []
    for lib_dir in sorted(data_dir.iterdir()):
        if not lib_dir.is_dir():
            continue
        # Prefer annotations.jsonl (has LLM descriptions) over declarations.jsonl
        jsonl = lib_dir / "annotations.jsonl"
        if not jsonl.exists():
            jsonl = lib_dir / "declarations.jsonl"
        if not jsonl.exists():
            continue
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    decls.append(json.loads(line))
    return decls


def make_embedding_text(decl: dict) -> str:
    """Create the text to embed for a declaration.

    If an annotation exists, use it. Otherwise, use name + type.
    """
    annotation = decl.get("annotation")
    if annotation:
        return f"{decl['name']}: {annotation}"
    # Fallback: name + type (truncated for very long types)
    type_str = decl["type"]
    if len(type_str) > 500:
        type_str = type_str[:500] + "..."
    return f"{decl['name']}: {type_str}"


def build_index(data_dir: Path | None = None, doc_prompt: bool = True) -> None:
    """Build FAISS index from all extracted declarations."""
    if data_dir is None:
        data_dir = DATA_DIR

    decls = load_declarations(data_dir)
    if not decls:
        console.print("[red]No declarations found. Run extraction first.[/red]")
        sys.exit(1)

    console.print(f"Loaded {len(decls)} declarations")

    # Create embedding texts
    texts = [make_embedding_text(d) for d in decls]

    # Load model and encode
    console.print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    console.print("Encoding declarations...")
    encode_kwargs: dict = dict(show_progress_bar=True, batch_size=8, normalize_embeddings=True)
    if doc_prompt:
        encode_kwargs["prompt"] = EMBEDDING_DOC_PROMPT
    embeddings = model.encode(texts, **encode_kwargs)
    embeddings = np.array(embeddings, dtype=np.float32)

    console.print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index (inner product = cosine similarity since normalized)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save index
    index_path = data_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    console.print(f"Saved FAISS index to {index_path}")

    # Save metadata (parallel array)
    metadata_path = data_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for decl in decls:
            f.write(json.dumps(decl, ensure_ascii=False) + "\n")
    console.print(f"Saved metadata to {metadata_path}")

    console.print(f"[green]Index built: {len(decls)} declarations, {embeddings.shape[1]}-dim embeddings[/green]")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build search index from extracted declarations")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-doc-prompt", action="store_true", help="Disable document-side instruction prompt")
    args = parser.parse_args()

    build_index(args.data_dir, doc_prompt=not args.no_doc_prompt)


if __name__ == "__main__":
    main()
