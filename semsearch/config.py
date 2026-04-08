from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory for extracted/annotated/embedded data
DATA_DIR = PROJECT_ROOT / "data"

# Coq paths
OPAM_PREFIX = Path.home() / ".opam" / "genproof"
COQ_LIB = OPAM_PREFIX / "lib" / "coq"
COQ_THEORIES = COQ_LIB / "theories"
COQ_USER_CONTRIB = COQ_LIB / "user-contrib"
COQTOP = OPAM_PREFIX / "bin" / "coqtop"

# Libraries to extract (name -> base path)
# Stdlib lives under theories/, everything else under user-contrib/
LIBRARIES: dict[str, Path] = {
    "Coq": COQ_THEORIES,
    "Bignums": COQ_USER_CONTRIB / "Bignums",
    "compcert": COQ_USER_CONTRIB / "compcert",
    "Flocq": COQ_USER_CONTRIB / "Flocq",
    "iris": COQ_USER_CONTRIB / "iris",
    "mathcomp": COQ_USER_CONTRIB / "mathcomp",
    "MenhirLib": COQ_USER_CONTRIB / "MenhirLib",
    "stdpp": COQ_USER_CONTRIB / "stdpp",
    "VST": COQ_USER_CONTRIB / "VST",
}

# Embedding model
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIM = 4096
EMBEDDING_QUERY_PROMPT = "Instruct: Given a Coq theorem prover query, retrieve relevant declarations\nQuery: "
EMBEDDING_DOC_PROMPT = None  # Qwen3-Embedding: instructions on queries only, documents plain

# Search defaults
DEFAULT_K = 10

# Server
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8089

# Annotation
ANNOTATION_MODEL = "claude-haiku-4-5-20251001"
ANNOTATION_BATCH_SIZE = 1000
