"""Search quality tests: check that expected results appear in top-k."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from semsearch.search import SearchEngine

K = 200  # Search depth to look for expected results

# (query, [(expected_name_substring, match_type)])
# match_type: "exact" means name must equal pattern, "substring" means name must contain pattern
TEST_CASES: list[tuple[str, list[str]]] = [
    ("Integer division", ["Z.div"]),
    ("Type representing finite maps", ["gmap", "GMap", "FMap", "Pmap"]),
    ("Finite mapping type", ["gmap", "GMap", "FMap", "Pmap"]),
    ("separation logic cancel vst", ["cancel_left", "cancel1_start", "cancel1_next"]),
    ("transitivity in separation logic in vst", ["derives_trans"]),
    ("mpred ext", ["pred_ext"]),
]


def find_rank(
    results: list, pattern: str
) -> tuple[int | None, str | None, float | None]:
    """Find the rank (1-indexed) of the first result whose name contains pattern.

    Prefers exact match, falls back to substring match.
    """
    # Exact match first
    for i, r in enumerate(results):
        if r.name == pattern:
            return i + 1, r.name, r.score
    # Substring match
    for i, r in enumerate(results):
        if pattern.lower() in r.name.lower():
            return i + 1, r.name, r.score
    return None, None, None


def main():
    engine = SearchEngine()

    all_results = []

    for query, expected_patterns in TEST_CASES:
        results = engine.search(query, k=K)

        print(f"=== {query!r} ===")
        for pattern in expected_patterns:
            rank, matched, score = find_rank(results, pattern)
            if rank:
                print(f"  {pattern}: rank {rank} (matched: {matched}, score: {score:.3f})")
            else:
                print(f"  {pattern}: NOT FOUND in top {K}")
            all_results.append(
                {
                    "query": query,
                    "expected": pattern,
                    "rank": rank,
                    "matched": matched,
                    "score": float(score) if score else None,
                }
            )
        print()

    # Summary
    found = sum(1 for r in all_results if r["rank"] is not None)
    total = len(all_results)
    found_ranks = [r["rank"] for r in all_results if r["rank"] is not None]
    avg_rank = sum(found_ranks) / max(len(found_ranks), 1)

    print(f"Summary: {found}/{total} found in top {K}, avg rank = {avg_rank:.1f}")
    print()

    # Breakdown by threshold
    for threshold in [5, 10, 20, 50]:
        in_top = sum(1 for r in found_ranks if r <= threshold)
        print(f"  In top {threshold:3d}: {in_top}/{total}")

    # Save results
    out_path = Path(__file__).parent.parent / "data" / "search_quality.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
