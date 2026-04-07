from pydantic import BaseModel


class Declaration(BaseModel):
    """A single Coq declaration extracted from a library."""

    name: str  # Fully qualified name, e.g. "Z.add_comm"
    type: str  # Type signature, e.g. "forall n m : Z, (n + m)%Z = (m + n)%Z"
    kind: str | None = None  # "Lemma", "Definition", "Inductive", etc.
    library: str  # Top-level library, e.g. "Coq" or "compcert"
    module: str  # Full module path, e.g. "Coq.ZArith.BinInt"
    source_file: str | None = None  # Path to .v file if available
    file_docstring: str | None = None
    decl_docstring: str | None = None
    is_opaque: bool | None = None
    arguments: str | None = None  # Arguments info from About


class AnnotatedDeclaration(Declaration):
    """A declaration with a natural language annotation."""

    annotation: str  # LLM-generated natural language description


class SearchResult(BaseModel):
    """A search result returned to the user."""

    name: str
    type: str
    kind: str | None = None
    library: str
    module: str
    annotation: str | None = None
    score: float
