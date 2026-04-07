# semsearch — Semantic Search for Coq

Semantic search engine for Coq theorem prover libraries.

## Pipeline

1. **Extract** — Dump all definitions, theorems, lemmas, etc. from compiled Coq libraries (.vo files) with their names and types
2. **Annotate** — Generate natural language descriptions of each declaration using an LLM
3. **Embed** — Build vector embeddings from the annotations
4. **Search** — KNN-based retrieval: query in natural language → find relevant Coq declarations

## Architecture

- Python project (3.14+, uv for deps)
- Coq 8.20.1 in opam switch `genproof` at `/home/vasa/.opam/genproof/`
- Target libraries: all deps of `/home/vasa/_work/genproof/liblzma-verification`
  - Coq stdlib, coq-bignums, coq-compcert, coq-flocq, coq-iris, coq-mathcomp-*, coq-menhirlib, coq-stdpp, coq-vst, coq-vst-ora, coq-vst-zlist
- Scale: ~100k declarations total, simple KNN is sufficient
- Must be extensible: add/remove libraries from the index

## Extraction

Uses `coqtop` in batch mode. Load a library, run `Search _` to get all names + types. Parse the structured output. Each declaration gets: fully qualified name, type signature, source library, kind (Lemma/Definition/Inductive/etc).

## Key directories

- `semsearch/` — Python package
- `data/` — extracted declarations, annotations, embeddings (gitignored)
