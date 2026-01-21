#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKOS 7-Star Mapping (SKOS-Plus)
===============================

This script implements a **7-star refinement layer** for SKOS mapping relations.

It produces a **property-only ontology (TBox)**:
- stars 1..5: five sub-properties of skos:relatedMatch as skosplus:* owl:ObjectProperty
- star 6:      skos:closeMatch (kept as SKOS property)
- star 7:      skos:exactMatch (kept as SKOS property)

Each mapping relation is annotated with a numeric degree:
- skosplus:degreeOfConnection (0..1)
computed via a normalised, saturating exponential function.

Internal logic documentation (merge-safe)
----------------------------------------
Earlier versions created an additional object-property hierarchy "perception band" with
phrase properties below it. This was confusing in Protégé after merging.

To keep the explanatory "perception band" idea WITHOUT polluting the object-property tree:
- skosplus:perceptionBand is an owl:AnnotationProperty (not an ObjectProperty)
- no rdfs:subPropertyOf links are created to phrase properties here
- the star relations are annotated with a phrase label as an internal justification

Inputs
------
- probly.csv : perceptions-of-probability dataset (wide or long format supported)

Outputs (filenames unchanged)
----------------------------
- skos_7star_mapping.ttl      : RDF/Turtle
- skos_7star_degrees.csv      : degrees table (stars 1..7)
- skos_7star_degree_plot.jpg  : 300 DPI line plot of star->degree

Namespace
---------
- skosplus: https://w3id.org/skos-plus/

Usage
-----
Run from within the folder (where probly.csv is located):

    python skos_7star.py

Optional arguments:

    python skos_7star.py --help
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD


# -----------------------------
# CONFIG
# -----------------------------

SKOSPLUS_NS = "https://w3id.org/skos-plus/"
PERCEPTIONS_REPO = "https://github.com/zonination/perceptions"

OUT_TTL = "skos_7star_mapping.ttl"
OUT_DEGREES_CSV = "skos_7star_degrees.csv"
OUT_PLOT_JPG = "skos_7star_degree_plot.jpg"

DEFAULT_K = 2.0
EXPORT_DPI = 300


# -----------------------------
# DEGREE FUNCTION
# -----------------------------


def degree_of_connection(star: int, k: float) -> float:
    """
    Normalised saturating exponential mapping from star level (1..7) to [0,1].

    d(s) = (1 - exp(-k*(s-1)/6)) / (1 - exp(-k))

    - s=1 => 0.0
    - s=7 => 1.0
    """
    if not (1 <= star <= 7):
        raise ValueError("star must be in [1..7]")
    if k <= 0:
        raise ValueError("k must be > 0")

    x = (star - 1) / 6.0
    numerator = 1.0 - math.exp(-k * x)
    denom = 1.0 - math.exp(-k)
    d = numerator / denom if denom != 0 else 0.0
    return max(0.0, min(1.0, d))


# -----------------------------
# IO: LOAD PROBLY.CSV
# -----------------------------


def load_probly_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load perceptions CSV and return a cleaned long-format dataframe with columns:
    - phrase (str)
    - probability (float in 0..1)

    Supports:
    1) Long format: columns 'phrase' and 'probability'
    2) Wide format: each column is a phrase; values in 0..100 or 0..1
    """
    df_raw = pd.read_csv(csv_path)
    cols_norm = [c.strip().lower() for c in df_raw.columns]

    # Case 1: long format
    if "phrase" in cols_norm and "probability" in cols_norm:
        df = df_raw.copy()
        phrase_col = df_raw.columns[cols_norm.index("phrase")]
        prob_col = df_raw.columns[cols_norm.index("probability")]
        df = df.rename(columns={phrase_col: "phrase", prob_col: "probability"})
        df["phrase"] = df["phrase"].astype(str).str.strip()
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
        df = df.dropna(subset=["phrase", "probability"]).copy()
    else:
        # Case 2: wide format
        df = df_raw.melt(var_name="phrase", value_name="probability")
        df["phrase"] = df["phrase"].astype(str).str.strip()
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
        df = df.dropna(subset=["phrase", "probability"]).copy()

    # Convert 0..100 -> 0..1 if needed
    if len(df) and df["probability"].max() > 1.0:
        df["probability"] = df["probability"] / 100.0

    # Keep only valid unit interval values
    df = df[(df["probability"] >= 0.0) & (df["probability"] <= 1.0)].copy()
    return df


def phrase_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Compute median probability per phrase (0..1)."""
    return (
        df.groupby("phrase", as_index=False)["probability"]
        .median()
        .rename(columns={"probability": "median_probability"})
        .sort_values("median_probability")
        .reset_index(drop=True)
    )


# -----------------------------
# 7-STAR DEFINITIONS
# -----------------------------


def mapping_definitions() -> Dict[int, Dict[str, str]]:
    """
    Define the 7 star levels, relation localnames, labels, and a suggested perceptions phrase.

    Stars:
      1..5 -> skosplus:* subproperties of skos:relatedMatch
      6    -> skos:closeMatch
      7    -> skos:exactMatch
    """
    return {
        1: {
            "localname": "relatedMatchVeryWeak",
            "label": "relatedMatch very weak (7-star level 1)",
            "phrase": "We Doubt",
        },
        2: {
            "localname": "relatedMatchWeak",
            "label": "relatedMatch weak (7-star level 2)",
            "phrase": "About Even",
        },
        3: {
            "localname": "relatedMatchModerate",
            "label": "relatedMatch moderate (7-star level 3)",
            "phrase": "Better Than Even",
            "phrase_alt": "Probably",
        },
        4: {
            "localname": "relatedMatchStrong",
            "label": "relatedMatch strong (7-star level 4)",
            "phrase": "Probable",
            "phrase_alt": "Likely",
        },
        5: {
            "localname": "relatedMatchVeryStrong",
            "label": "relatedMatch very strong (7-star level 5)",
            "phrase": "Very Good Chance",
        },
        6: {
            "localname": "closeMatch",
            "label": "SKOS closeMatch (7-star level 6)",
            "phrase": "Highly Likely",
            "phrase_alt": "Very Good Chance",
        },
        7: {
            "localname": "exactMatch",
            "label": "SKOS exactMatch (7-star level 7)",
            "phrase": "Almost Certainly",
            "phrase_alt": "Highly Likely",
        },
    }


def find_phrase_value(medians: pd.DataFrame, phrase: str) -> float | None:
    """Look up the median for a phrase (case-insensitive)."""
    hit = medians.loc[
        medians["phrase"].str.casefold() == phrase.casefold(), "median_probability"
    ]
    if len(hit) == 0:
        return None
    return float(hit.iloc[0])


def choose_phrase_for_star(
    medians: pd.DataFrame, phrase: str | None, phrase_alt: str | None
) -> Tuple[str | None, float | None]:
    """
    Pick the best phrase that exists in the CSV medians.
    Returns (chosen_phrase, median_value).
    """
    if phrase:
        v = find_phrase_value(medians, phrase)
        if v is not None:
            return phrase, v
    if phrase_alt:
        v = find_phrase_value(medians, phrase_alt)
        if v is not None:
            return phrase_alt, v
    return phrase, None


# -----------------------------
# RDF HELPERS
# -----------------------------


def bind_common_prefixes(g: Graph) -> Namespace:
    """Bind common prefixes for stable Turtle output and return the SKOSPLUS namespace."""
    skosplus = Namespace(SKOSPLUS_NS)
    g.bind("skos", SKOS)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    g.bind("dct", DCTERMS)
    g.bind("skosplus", skosplus)
    return skosplus


def ensure_skos_backbone(g: Graph) -> None:
    """
    Make the SKOS mapping backbone explicit for Protégé:
    - declare skos:Concept as OWL class
    - declare mapping properties as owl:ObjectProperty with domain/range skos:Concept
    """
    g.add((SKOS.Concept, RDF.type, OWL.Class))
    for p in (SKOS.relatedMatch, SKOS.closeMatch, SKOS.exactMatch):
        g.add((p, RDF.type, OWL.ObjectProperty))
        g.add((p, RDFS.domain, SKOS.Concept))
        g.add((p, RDFS.range, SKOS.Concept))


def declare_annotation_property(
    g: Graph, prop, label_en: str, comment_en: str | None = None
) -> None:
    """Declare an annotation property with label/comment."""
    g.add((prop, RDF.type, OWL.AnnotationProperty))
    g.add((prop, RDFS.label, Literal(label_en, lang="en")))
    if comment_en:
        g.add((prop, RDFS.comment, Literal(comment_en, lang="en")))


# -----------------------------
# RDF BUILD
# -----------------------------


def build_rdf(medians: pd.DataFrame, k: float, outdir: Path) -> pd.DataFrame:
    """
    Build the RDF graph and write:
    - OUT_TTL
    - OUT_DEGREES_CSV

    Returns the degrees table as DataFrame.
    """
    g = Graph()
    skosplus = bind_common_prefixes(g)
    ensure_skos_backbone(g)

    # Annotation properties (stable across scripts)
    DEG = skosplus.degreeOfConnection
    STAR = skosplus.starLevel
    MED = skosplus.medianPerceivedProbability
    PBAND = skosplus.perceptionBand  # merge-safe: annotation only

    declare_annotation_property(
        g,
        DEG,
        "degree of connection (0–1)",
        "Numeric degree of connection for 7-star mapping relations; computed via a normalised saturating exponential function.",
    )
    declare_annotation_property(
        g,
        STAR,
        "7-star level",
        "Discrete 7-star mapping level (1..7) used by the SKOS-Plus extension.",
    )
    declare_annotation_property(
        g,
        MED,
        "median perceived probability",
        "Median perceived probability for a perceptions phrase (derived from zonination/perceptions).",
    )
    declare_annotation_property(
        g,
        PBAND,
        "perception band",
        "Internal explanatory link to a perceptions phrase/band used to justify a 7-star level (annotation only; not part of the object-property hierarchy).",
    )
    g.add((PBAND, DCTERMS.source, Literal(PERCEPTIONS_REPO)))

    defs = mapping_definitions()
    rows: List[Dict[str, object]] = []

    for star, info in defs.items():
        label = info["label"]
        phrase = info.get("phrase")
        phrase_alt = info.get("phrase_alt")
        chosen_phrase, chosen_median = choose_phrase_for_star(
            medians, phrase, phrase_alt
        )

        # Determine the mapping relation URI
        if star <= 5:
            rel_uri = skosplus[info["localname"]]
            g.add((rel_uri, RDF.type, OWL.ObjectProperty))
            g.add((rel_uri, RDFS.subPropertyOf, SKOS.relatedMatch))
            g.add((rel_uri, RDFS.label, Literal(label, lang="en")))
            g.add((rel_uri, RDFS.domain, SKOS.Concept))
            g.add((rel_uri, RDFS.range, SKOS.Concept))
        elif star == 6:
            rel_uri = SKOS.closeMatch
            g.add((rel_uri, RDFS.label, Literal(label, lang="en")))
        else:
            rel_uri = SKOS.exactMatch
            g.add((rel_uri, RDFS.label, Literal(label, lang="en")))

        # Star + degree annotations
        d = degree_of_connection(star, k)
        g.add((rel_uri, DEG, Literal(d, datatype=XSD.decimal)))
        g.add((rel_uri, STAR, Literal(star, datatype=XSD.integer)))

        # Internal explanatory link (annotation only)
        if chosen_phrase:
            g.add((rel_uri, PBAND, Literal(chosen_phrase, lang="en")))
        if chosen_median is not None:
            g.add((rel_uri, MED, Literal(float(chosen_median), datatype=XSD.decimal)))

        rows.append(
            {
                "star": star,
                "relation_uri": str(rel_uri),
                "label": label,
                "degree_of_connection": round(d, 6),
                "k": k,
                "perception_phrase": chosen_phrase,
                "perception_median": (
                    None if chosen_median is None else round(float(chosen_median), 6)
                ),
            }
        )

    degrees = pd.DataFrame(rows).sort_values("star").reset_index(drop=True)

    outdir.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(outdir / OUT_TTL), format="turtle")
    degrees.to_csv(outdir / OUT_DEGREES_CSV, index=False, encoding="utf-8")

    return degrees


# -----------------------------
# PLOT
# -----------------------------


def plot_degrees(degrees: pd.DataFrame, outdir: Path) -> Path:
    """Create a line plot (star -> degree) and write a 300 DPI JPG."""
    x = degrees["star"].to_numpy()
    y = degrees["degree_of_connection"].to_numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker="o")
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.xlabel("7-Star Mapping Level")
    plt.ylabel("Degree of Connection (0–1)")
    plt.title("Degree of Connection for SKOS 7-Star Mapping")

    out_file = outdir / OUT_PLOT_JPG
    plt.savefig(out_file, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close()
    return out_file


# -----------------------------
# MAIN
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SKOS 7-star mapping RDF + degree plot from probly.csv"
    )
    parser.add_argument(
        "--csv",
        default="probly.csv",
        help="Input CSV (wide or long). Default: probly.csv",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_K,
        help=f"Exponential steepness parameter k (>0). Default: {DEFAULT_K}",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory (relative to this script). Default: current folder",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    csv_path = (script_dir / args.csv).resolve()
    outdir = (script_dir / args.outdir).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_probly_csv(csv_path)
    medians = phrase_medians(df)

    degrees = build_rdf(medians, args.k, outdir)
    plot_file = plot_degrees(degrees, outdir)

    # Consistent summary output
    print("Done.")
    print(f"- CSV file:      {csv_path}")
    print(f"- Loaded rows:   {len(df):,}")
    print(f"- Unique phrases:{len(medians):,}")
    print(f"- RDF (Turtle):  {outdir / OUT_TTL}")
    print(f"- Degrees CSV:   {outdir / OUT_DEGREES_CSV}")
    print(f"- Plot JPG:      {plot_file}")
    print()
    print("Degrees (star -> degree):")
    for _, row in degrees.iterrows():
        print(f"  {int(row['star'])} -> {row['degree_of_connection']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
