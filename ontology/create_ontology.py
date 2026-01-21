#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_ontology.py
------------------
Merges the three Turtle files into a single consolidated Turtle file.

Folder layout (as requested):
root/
  ontology/
    create_ontology.py   <-- this script
  skos/
    skos_minimal_degrees.ttl
  skos_7star/
    skos_7star_mapping.ttl
  skos_perceptions/
    skos_perceptions_mapping.ttl

Output:
root/ontology/merged_ontology.ttl

Notes:
- Uses rdflib to parse each TTL and add triples to a single Graph.
- Keeps namespaces/prefixes as much as possible by re-binding common ones.
- Optionally checks for URI collisions where the same subject+predicate has
  multiple distinct objects (can happen when modules disagree).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Tuple, Dict, Set

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS, DCTERMS


def repo_root_from_this_file() -> Path:
    """Assumes this script is located at root/ontology/create_ontology.py."""
    return Path(__file__).resolve().parents[1]


def ttl_paths(root: Path) -> Tuple[Path, Path, Path]:
    """Return the three module TTL paths."""
    p1 = root / "skos" / "skos_minimal_degrees.ttl"
    p2 = root / "skos_7star" / "skos_7star_mapping.ttl"
    p3 = root / "skos_perceptions" / "skos_perceptions_mapping.ttl"
    return p1, p2, p3


def bind_common_prefixes(g: Graph) -> None:
    """Bind common prefixes for a cleaner Turtle serialisation."""
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    g.bind("skos", SKOS)
    g.bind("dct", DCTERMS)
    # Your project prefix (used across modules)
    g.bind("skosplus", "https://w3id.org/skos-plus/")


def parse_into(target: Graph, ttl_file: Path) -> int:
    """Parse ttl_file into target graph; returns number of triples added."""
    before = len(target)
    target.parse(str(ttl_file), format="turtle")
    return len(target) - before


def find_potential_conflicts(g: Graph) -> Dict[Tuple[str, str], Set[str]]:
    """
    Find potential conflicts where (subject, predicate) maps to multiple objects.
    This is NOT always an error (e.g., multiple labels in different languages),
    but it helps to spot accidental re-definitions across modules.

    Returns: dict keyed by (s, p) -> set(objects)
    """
    conflicts: Dict[Tuple[str, str], Set[str]] = {}
    spo_map: Dict[Tuple[str, str], Set[str]] = {}

    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(p, URIRef):
            key = (str(s), str(p))
            spo_map.setdefault(key, set()).add(str(o))

    for key, objs in spo_map.items():
        if len(objs) > 1:
            conflicts[key] = objs

    return conflicts


def main() -> None:
    root = repo_root_from_this_file()
    in1, in2, in3 = ttl_paths(root)

    missing = [p for p in (in1, in2, in3) if not p.exists()]
    if missing:
        print("ERROR: Missing input TTL file(s):", file=sys.stderr)
        for p in missing:
            print(f" - {p}", file=sys.stderr)
        sys.exit(1)

    out_file = (root / "ontology" / "merged_ontology.ttl").resolve()

    merged = Graph()
    bind_common_prefixes(merged)

    added1 = parse_into(merged, in1)
    added2 = parse_into(merged, in2)
    added3 = parse_into(merged, in3)

    # Optional: conflict scan (prints a small summary; does not fail the build)
    conflicts = find_potential_conflicts(merged)

    merged.serialize(destination=str(out_file), format="turtle")

    print("Merged ontology created.")
    print(f"Repo root: {root}")
    print(f"Inputs:")
    print(f" - {in1} (+{added1} triples)")
    print(f" - {in2} (+{added2} triples)")
    print(f" - {in3} (+{added3} triples)")
    print(f"Output:")
    print(f" - {out_file}")
    print(f"Total triples: {len(merged)}")

    # Keep the conflict report short; it can be large.
    # We show only a handful of examples to help debugging.
    if conflicts:
        print("\nPotential (subject,predicate)->multiple objects situations detected:")
        print(f"Count: {len(conflicts)}")
        shown = 0
        for (s, p), objs in conflicts.items():
            # Skip some common "benign" predicates if you like (commented out):
            # if p in (str(RDFS.label), str(RDFS.comment), str(DCTERMS.source)):
            #     continue
            print(f"\n- {s}\n  {p}")
            for o in sorted(list(objs))[:10]:
                print(f"   -> {o}")
            if len(objs) > 10:
                print(f"   ... (+{len(objs)-10} more)")
            shown += 1
            if shown >= 10:
                print("\n(Showing first 10 only.)")
                break
    else:
        print("\nNo (subject,predicate) multi-object situations detected.")

    print("\nDone.")


if __name__ == "__main__":
    main()
