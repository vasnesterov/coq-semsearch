"""Annotate Coq declarations with natural language descriptions using Claude Batch API."""

import hashlib
import json
import sys
import time
from pathlib import Path

from rich.console import Console

from semsearch.config import ANNOTATION_BATCH_SIZE, ANNOTATION_MODEL, DATA_DIR

console = Console()

SYSTEM_PROMPT = """You are an expert in Coq theorem prover and formal verification. Given a Coq declaration (name and type signature), write a concise natural language description (1-2 sentences) of what it states or computes. Be precise but accessible. Use standard mathematical terminology where appropriate.

Examples:
- For "Z.add_comm: forall n m : Z, (n + m)%Z = (m + n)%Z", write: "Addition of integers is commutative: for any integers n and m, n + m equals m + n."
- For "List.app_nil_r: forall (A : Type) (l : list A), l ++ [] = l", write: "Appending an empty list to the right of any list yields the original list (right identity of list concatenation)."
- For "Z.mul: Z -> Z -> Z", write: "Multiplication function on integers."
"""

USER_TEMPLATE = """Coq declaration:
Name: {name}
Type: {type}
Library: {library}
{docstring_line}

Write a concise (1-2 sentence) natural language description."""


def decl_hash(decl: dict) -> str:
    """Content hash for a declaration (for caching)."""
    key = f"{decl['name']}:{decl['type']}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def make_user_message(decl: dict) -> str:
    doc = decl.get("decl_docstring") or decl.get("file_docstring")
    docstring_line = f"Docstring: {doc}" if doc else ""
    type_str = decl["type"]
    if len(type_str) > 2000:
        type_str = type_str[:2000] + "..."
    return USER_TEMPLATE.format(
        name=decl["name"],
        type=type_str,
        library=decl["library"],
        docstring_line=docstring_line,
    )


def create_batch_requests(decls: list[dict]) -> list[dict]:
    """Create batch API request objects for a list of declarations."""
    requests = []
    for decl in decls:
        custom_id = decl_hash(decl)
        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": ANNOTATION_MODEL,
                "max_tokens": 200,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": make_user_message(decl)}],
            },
        })
    return requests


def annotate_library(lib_dir: Path, limit: int | None = None) -> int:
    """Annotate all declarations in a library directory.

    Returns number of annotations added.
    """
    import random

    import anthropic

    decls_file = lib_dir / "declarations.jsonl"
    annotations_file = lib_dir / "annotations.jsonl"

    if not decls_file.exists():
        console.print(f"  [yellow]No declarations found in {lib_dir}[/yellow]")
        return 0

    # Load declarations
    decls = []
    with open(decls_file) as f:
        for line in f:
            line = line.strip()
            if line:
                decls.append(json.loads(line))

    # Load existing annotations (if any) for caching
    existing: dict[str, str] = {}  # hash -> annotation
    if annotations_file.exists():
        with open(annotations_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    h = decl_hash(d)
                    if d.get("annotation"):
                        existing[h] = d["annotation"]

    # Find declarations that need annotation
    to_annotate = []
    for decl in decls:
        h = decl_hash(decl)
        if h not in existing:
            to_annotate.append(decl)

    # Random subset if limit is set
    if limit is not None and len(to_annotate) > limit:
        random.shuffle(to_annotate)
        to_annotate = to_annotate[:limit]
        console.print(f"  Sampled {limit} random declarations to annotate")

    if not to_annotate:
        console.print(f"  All {len(decls)} declarations already annotated")
        return 0

    console.print(f"  {len(to_annotate)} declarations need annotation ({len(existing)} cached)")

    client = anthropic.Anthropic()

    # Process in batches
    for batch_start in range(0, len(to_annotate), ANNOTATION_BATCH_SIZE):
        batch = to_annotate[batch_start : batch_start + ANNOTATION_BATCH_SIZE]
        batch_num = batch_start // ANNOTATION_BATCH_SIZE + 1
        total_batches = (len(to_annotate) + ANNOTATION_BATCH_SIZE - 1) // ANNOTATION_BATCH_SIZE
        console.print(f"  Submitting batch {batch_num}/{total_batches} ({len(batch)} items)...")

        requests = create_batch_requests(batch)

        # Create batch
        message_batch = client.messages.batches.create(requests=requests)
        batch_id = message_batch.id
        console.print(f"  Batch {batch_id} created, waiting for completion...")

        # Poll for completion
        while True:
            batch_status = client.messages.batches.retrieve(batch_id)
            if batch_status.processing_status == "ended":
                break
            time.sleep(10)

        # Collect results
        failed = 0
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            if result.result.type == "succeeded":
                text = result.result.message.content[0].text
                existing[custom_id] = text
            else:
                failed += 1

        console.print(f"  Batch {batch_num} done: {len(batch) - failed} ok, {failed} failed")

        # Save after each batch so we can safely resume if interrupted
        with open(annotations_file, "w") as f:
            for decl in decls:
                h = decl_hash(decl)
                annotated = dict(decl)
                annotated["annotation"] = existing.get(h)
                f.write(json.dumps(annotated, ensure_ascii=False) + "\n")

    new_count = sum(1 for d in decls if decl_hash(d) in existing)
    console.print(f"  [green]Annotated {new_count}/{len(decls)} declarations[/green]")
    return len(to_annotate)


def main() -> None:
    import argparse

    from semsearch.config import LIBRARIES

    parser = argparse.ArgumentParser(description="Annotate Coq declarations with LLM")
    parser.add_argument(
        "--library",
        "-l",
        help="Annotate only this library",
        choices=list(LIBRARIES.keys()),
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Annotate only N random declarations (for testing)",
    )
    args = parser.parse_args()

    total = 0
    if args.library:
        lib_dirs = [(args.library, DATA_DIR / args.library)]
    else:
        lib_dirs = [(name, DATA_DIR / name) for name in LIBRARIES]

    for lib_name, lib_dir in lib_dirs:
        if not lib_dir.exists():
            continue
        console.print(f"[bold]Annotating {lib_name}...[/bold]")
        count = annotate_library(lib_dir, limit=args.limit)
        total += count

    console.print(f"\n[bold green]Annotated {total} new declarations[/bold green]")


if __name__ == "__main__":
    main()
