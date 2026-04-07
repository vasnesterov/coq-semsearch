"""Extract declarations from compiled Coq libraries."""

import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from semsearch.config import COQ_THEORIES, COQ_USER_CONTRIB, COQTOP, DATA_DIR, LIBRARIES
from semsearch.models import Declaration

console = Console()


def vo_to_module(vo_path: Path, base_dir: Path, library: str) -> str:
    """Convert a .vo file path to a Coq module path.

    For stdlib: base_dir is theories/, module is Coq.<relative path>
    For user-contrib: base_dir is user-contrib/Lib/, module is Lib.<relative path>
    """
    rel = vo_path.relative_to(base_dir)
    parts = rel.with_suffix("").parts
    if library == "Coq":
        return "Coq." + ".".join(parts)
    else:
        return library + "." + ".".join(parts)


def find_modules(library: str, base_dir: Path) -> list[tuple[str, Path]]:
    """Find all .vo files in a library and return (module_path, vo_path) pairs."""
    vo_files = sorted(base_dir.rglob("*.vo"))
    result = []
    for vo in vo_files:
        try:
            module = vo_to_module(vo, base_dir, library)
            result.append((module, vo))
        except ValueError:
            continue
    return result


def extract_module(module: str, coqtop: str = str(COQTOP)) -> list[tuple[str, str]]:
    """Extract all declarations from a single Coq module.

    Returns list of (name, type) pairs.
    """
    script = f"Require Import {module}.\nSearch _ inside {module}.\n"
    try:
        result = subprocess.run(
            [coqtop, "-q"],
            input=script,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return []

    output = result.stdout
    return parse_search_output(output)


def parse_search_output(output: str) -> list[tuple[str, str]]:
    """Parse coqtop Search output into (name, type) pairs.

    Output format:
    - Normal: `name: type`
    - Multi-line: `name:` followed by indented continuation lines
    - Noise: lines starting with Welcome, Coq <, [Loading, etc.
    """
    declarations: list[tuple[str, str]] = []
    current_name: str | None = None
    current_type_parts: list[str] = []

    for line in output.splitlines():
        # Skip noise
        if (
            not line
            or line.startswith("Welcome")
            or line.startswith("Skipping")
            or line.startswith("[Loading")
            or line.startswith("Coq <")
            or line.startswith("(use ")
            or line.startswith("Toplevel")
            or line.startswith("> ")
            or line.startswith("Error:")
            or line.startswith("Warning:")
        ):
            continue

        # Continuation line (starts with whitespace)
        if line[0] in (" ", "\t"):
            if current_name is not None:
                current_type_parts.append(line.strip())
            continue

        # New declaration line: name: type
        # First, save previous declaration
        if current_name is not None:
            type_str = " ".join(current_type_parts)
            declarations.append((current_name, type_str))

        # Parse new declaration
        colon_idx = line.find(":")
        if colon_idx > 0:
            current_name = line[:colon_idx].strip()
            type_rest = line[colon_idx + 1 :].strip()
            current_type_parts = [type_rest] if type_rest else []
        else:
            current_name = None
            current_type_parts = []

    # Don't forget the last one
    if current_name is not None:
        type_str = " ".join(current_type_parts)
        declarations.append((current_name, type_str))

    return declarations


def parse_v_file_docstrings(v_path: Path) -> tuple[str | None, dict[str, str]]:
    """Parse a .v file to extract docstrings.

    Returns (file_docstring, {decl_name: docstring}).
    """
    if not v_path.exists():
        return None, {}

    try:
        text = v_path.read_text(errors="replace")
    except Exception:
        return None, {}

    file_doc = None
    decl_docs: dict[str, str] = {}

    # Extract all doc comments: (** ... *)
    doc_pattern = re.compile(r"\(\*\*\s(.*?)\*\)", re.DOTALL)
    # Declaration patterns
    decl_pattern = re.compile(
        r"^(?:Theorem|Lemma|Definition|Fixpoint|CoFixpoint|Inductive|CoInductive|"
        r"Record|Structure|Class|Instance|Axiom|Parameter|Hypothesis|Variable|"
        r"Coercion|Canonical|Notation|Program\s+\w+|#\[global\]\s+Instance|"
        r"Fact|Remark|Corollary|Proposition|Property|Example)\s+(\w+)",
        re.MULTILINE,
    )

    # Find the first doc comment as file docstring
    first_doc = doc_pattern.search(text)
    if first_doc and first_doc.start() < 200:  # Near start of file
        file_doc = first_doc.group(1).strip()

    # For each declaration, look for a doc comment immediately before it
    for decl_match in decl_pattern.finditer(text):
        decl_name = decl_match.group(1)
        decl_start = decl_match.start()

        # Look backwards for a doc comment ending just before this declaration
        preceding = text[max(0, decl_start - 500) : decl_start].rstrip()
        doc_matches = list(doc_pattern.finditer(preceding))
        if doc_matches:
            last_doc = doc_matches[-1]
            # Check it's close to the declaration (only whitespace between)
            between = preceding[last_doc.end() :].strip()
            if not between:  # Only whitespace between doc and decl
                decl_docs[decl_name] = last_doc.group(1).strip()

    return file_doc, decl_docs


def extract_library(
    library: str,
    base_dir: Path,
    output_dir: Path,
    max_workers: int = 8,
) -> int:
    """Extract all declarations from a library.

    Returns number of declarations extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "declarations.jsonl"

    modules = find_modules(library, base_dir)
    if not modules:
        console.print(f"  [yellow]No modules found for {library}[/yellow]")
        return 0

    console.print(f"  Found {len(modules)} modules in {library}")

    # Extract declarations from all modules in parallel
    all_decls: dict[str, Declaration] = {}  # deduplicate by name

    # Build a map of v_file -> (file_docstring, decl_docs) for docstring lookup
    v_file_cache: dict[Path, tuple[str | None, dict[str, str]]] = {}

    with Progress(console=console) as progress:
        task = progress.add_task(f"  Extracting {library}", total=len(modules))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for module_path, vo_path in modules:
                future = executor.submit(extract_module, module_path)
                futures[future] = (module_path, vo_path)

            for future in as_completed(futures):
                module_path, vo_path = futures[future]
                progress.advance(task)

                try:
                    decls = future.result()
                except Exception as e:
                    console.print(f"  [red]Error extracting {module_path}: {e}[/red]")
                    continue

                # Find corresponding .v file for docstrings
                v_path = vo_path.with_suffix(".v")
                if v_path not in v_file_cache:
                    v_file_cache[v_path] = parse_v_file_docstrings(v_path)
                file_doc, decl_docs = v_file_cache[v_path]

                for name, type_str in decls:
                    if name in all_decls:
                        continue  # deduplicate

                    # Extract short name for docstring lookup
                    short_name = name.rsplit(".", 1)[-1] if "." in name else name

                    all_decls[name] = Declaration(
                        name=name,
                        type=type_str,
                        library=library,
                        module=module_path,
                        source_file=str(v_path) if v_path.exists() else None,
                        file_docstring=file_doc,
                        decl_docstring=decl_docs.get(short_name),
                    )

    # Write output
    with open(out_file, "w") as f:
        for decl in sorted(all_decls.values(), key=lambda d: d.name):
            f.write(decl.model_dump_json() + "\n")

    count = len(all_decls)
    console.print(f"  [green]Extracted {count} declarations from {library}[/green]")
    return count


def main() -> None:
    """Extract declarations from all configured libraries."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract Coq declarations")
    parser.add_argument(
        "--library",
        "-l",
        help="Extract only this library (default: all)",
        choices=list(LIBRARIES.keys()),
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=8,
        help="Max parallel coqtop processes",
    )
    args = parser.parse_args()

    libraries = {args.library: LIBRARIES[args.library]} if args.library else LIBRARIES

    total = 0
    for lib_name, lib_path in libraries.items():
        if not lib_path.exists():
            console.print(f"[yellow]Skipping {lib_name}: {lib_path} not found[/yellow]")
            continue
        console.print(f"[bold]Extracting {lib_name}...[/bold]")
        count = extract_library(lib_name, lib_path, DATA_DIR / lib_name, args.workers)
        total += count

    console.print(f"\n[bold green]Total: {total} declarations extracted[/bold green]")


if __name__ == "__main__":
    main()
