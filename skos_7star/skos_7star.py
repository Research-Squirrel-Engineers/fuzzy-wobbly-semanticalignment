#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKOS 7-Star Mapping + Perceptions-of-Probability mapping (SKOS-Plus)
-------------------------------------------------------------------
- reads:  probly.csv (must be in same folder as this script)
- builds: an RDF graph (rdflib) with ONLY property hierarchies (no individuals):
    * 5 subproperties of skos:relatedMatch (stars 1..5) as skosplus:* object properties
    * skos:closeMatch (star 6) and skos:exactMatch (star 7) kept as SKOS object properties
    * introduces skos:Concept as a class (TBox only; no individuals)
    * sets rdfs:domain/rdfs:range for mapping-related properties to skos:Concept

IMPORTANT CHANGE (for merge consistency)
---------------------------------------
The previous version created an additional object-property hierarchy:
    skosplus:perceptionBand
        └─ skosplus:perceptions_<phrase>

This caused Protégé to show a second "ordering" of properties in the merged ontology.

To keep the explanatory "perception band" idea WITHOUT polluting the object-property tree:
- skosplus:perceptionBand is now an owl:AnnotationProperty (not an ObjectProperty)
- No rdfs:subPropertyOf links are created between mapping relations and perception phrase properties
- The internal/explanatory link is kept as an annotation on each star relation:
    skosplus:perceptionBand "We Doubt"@en
(or whichever phrase is used for that star)

This keeps:
- the merged ontology clean (SKOS-first hierarchy)
- the 7-star logic documented in the same TTL for working-paper explanation

- writes (FILENAMES UNCHANGED):
    * skos_7star_mapping.ttl      (Turtle)
    * skos_7star_degrees.csv      (table)
    * skos_7star_degree_plot.jpg  (300 DPI line chart)

Prefix/URI rules implemented
----------------------------
- Only one project prefix is used for custom terms:
    @prefix skosplus: <https://w3id.org/skos-plus/> .

Source
------
Perceptions phrases & distributions:
https://github.com/zonination/perceptions
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS, DCTERMS


# -----------------------------
# CONFIG
# -----------------------------

SKOSPLUS_NS = "https://w3id.org/skos-plus/"
PERCEPTIONS_REPO = "https://github.com/zonination/perceptions"

OUT_TTL = "skos_7star_mapping.ttl"
OUT_DEGREES_CSV = "skos_7star_degrees.csv"
OUT_PLOT_JPG = "skos_7star_degree_plot.jpg"

DEFAULT_K = 2.0


# -----------------------------
# Degree function (exponential)
# -----------------------------
def degree_of_connection(star: int, k: float) -> float:
    """
    Normalised saturating exponential from 0..1 for star in 1..7.

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
# Load & summarise probly.csv
# -----------------------------
def load_probly_csv(csv_path: Path) -> pd.DataFrame:
    """
    Loads the perceptions CSV and returns a cleaned dataframe with:
    - phrase (str)
    - probability (float 0..1)

    Supports:
    1) Long format with columns: phrase, probability
    2) Wide format where each column is a phrase and rows contain numeric values (0..100 or 0..1)
    """
    df_raw = pd.read_csv(csv_path)
    cols_norm = [c.strip().lower() for c in df_raw.columns]

    # Case 1: Long format
    if "phrase" in cols_norm and "probability" in cols_norm:
        df = df_raw.copy()
        phrase_col = df_raw.columns[cols_norm.index("phrase")]
        prob_col = df_raw.columns[cols_norm.index("probability")]
        df = df.rename(columns={phrase_col: "phrase", prob_col: "probability"})
        df["phrase"] = df["phrase"].astype(str).str.strip()
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
        df = df.dropna(subset=["phrase", "probability"]).copy()

    # Case 2: Wide format
    else:
        df = df_raw.melt(var_name="phrase", value_name="probability")
        df["phrase"] = df["phrase"].astype(str).str.strip()
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
        df = df.dropna(subset=["phrase", "probability"]).copy()

    # If values look like 0..100, convert to 0..1
    if df["probability"].max() > 1.0:
        df["probability"] = df["probability"] / 100.0

    df = df[(df["probability"] >= 0.0) & (df["probability"] <= 1.0)].copy()
    return df


def phrase_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe with median probability per phrase."""
    return (
        df.groupby("phrase", as_index=False)["probability"]
        .median()
        .rename(columns={"probability": "median_probability"})
        .sort_values("median_probability")
    )


# -----------------------------
# 7-star mapping definitions
# -----------------------------
def mapping_definitions() -> Dict[int, Dict[str, str]]:
    """
    Defines the 7 star levels, relation localnames, labels, and phrase mapping.

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
    hit = medians.loc[
        medians["phrase"].str.casefold() == phrase.casefold(), "median_probability"
    ]
    if len(hit) == 0:
        return None
    return float(hit.iloc[0])


def _choose_phrase_for_star(
    medians: pd.DataFrame, phrase: str | None, phrase_alt: str | None
) -> tuple[str | None, float | None]:
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
# RDF building (property-only model)
# -----------------------------
def build_rdf(
    medians: pd.DataFrame,
    k: float,
    out_path: Path,
) -> Tuple[Graph, pd.DataFrame]:
    SKOSPLUS = Namespace(SKOSPLUS_NS)

    g = Graph()
    g.bind("skos", SKOS)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("dct", DCTERMS)
    g.bind("xsd", XSD)
    g.bind("skosplus", SKOSPLUS)

    # --- Classes (TBox only; no individuals) ---
    g.add((SKOS.Concept, RDF.type, OWL.Class))

    # --- Ensure SKOS match properties are treated as object properties in this ontology view ---
    for p in (SKOS.relatedMatch, SKOS.closeMatch, SKOS.exactMatch):
        g.add((p, RDF.type, OWL.ObjectProperty))
        g.add((p, RDFS.domain, SKOS.Concept))
        g.add((p, RDFS.range, SKOS.Concept))

    # --- Annotation properties (avoid OWL2 punning/Individuals in Protégé) ---
    DEG = SKOSPLUS.degreeOfConnection
    STAR = SKOSPLUS.starLevel
    MED = SKOSPLUS.medianPerceivedProbability
    PBAND = SKOSPLUS.perceptionBand  # NOW: annotation property

    for ap, label, comment in [
        (
            DEG,
            "degree of connection (0–1)",
            "Numeric degree of connection for 7-star mapping relations; computed via a normalised saturating exponential function.",
        ),
        (
            STAR,
            "7-star level",
            "Discrete 7-star mapping level (1..7) used by the SKOS-Plus extension.",
        ),
        (
            MED,
            "median perceived probability",
            "Median perceived probability for a perceptions phrase (derived from zonination/perceptions).",
        ),
        (
            PBAND,
            "perception band",
            "Internal explanatory link to a perceptions phrase/band used to justify a 7-star level (annotation only; not part of the object-property hierarchy).",
        ),
    ]:
        g.add((ap, RDF.type, OWL.AnnotationProperty))
        g.add((ap, RDFS.label, Literal(label, lang="en")))
        g.add((ap, RDFS.comment, Literal(comment, lang="en")))

    g.add((PBAND, DCTERMS.source, Literal(PERCEPTIONS_REPO)))

    # --- Build 7-star relations ---
    defs = mapping_definitions()
    degrees_rows: List[Dict[str, object]] = []

    for star, info in defs.items():
        rel_local = info["localname"]
        label = info["label"]
        phrase = info.get("phrase")
        phrase_alt = info.get("phrase_alt")

        chosen_phrase, chosen_median = _choose_phrase_for_star(
            medians, phrase, phrase_alt
        )

        # Determine the mapping relation URI
        if star <= 5:
            rel_uri = SKOSPLUS[rel_local]
            g.add((rel_uri, RDF.type, OWL.ObjectProperty))
            g.add((rel_uri, RDFS.subPropertyOf, SKOS.relatedMatch))
            g.add((rel_uri, RDFS.label, Literal(label, lang="en")))
            g.add((rel_uri, RDFS.domain, SKOS.Concept))
            g.add((rel_uri, RDFS.range, SKOS.Concept))

        elif star == 6:
            rel_uri = SKOS.closeMatch
            g.add((rel_uri, RDFS.label, Literal(label, lang="en")))

        else:  # star == 7
            rel_uri = SKOS.exactMatch
            g.add((rel_uri, RDFS.label, Literal(label, lang="en")))

        # Add star + degree as annotations to the relation
        d = degree_of_connection(star, k)
        g.add((rel_uri, DEG, Literal(d, datatype=XSD.decimal)))
        g.add((rel_uri, STAR, Literal(star, datatype=XSD.integer)))

        # Internal explanatory link (annotation only)
        if chosen_phrase:
            g.add((rel_uri, PBAND, Literal(chosen_phrase, lang="en")))
        if chosen_median is not None:
            # optional: keep the median as annotation for documentation
            g.add((rel_uri, MED, Literal(float(chosen_median), datatype=XSD.decimal)))

        degrees_rows.append(
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

    degrees_df = pd.DataFrame(degrees_rows).sort_values("star")

    ttl_path = out_path / OUT_TTL
    g.serialize(destination=str(ttl_path), format="turtle")

    degrees_csv = out_path / OUT_DEGREES_CSV
    degrees_df.to_csv(degrees_csv, index=False, encoding="utf-8")

    return g, degrees_df


# -----------------------------
# Plot
# -----------------------------
def plot_degrees(degrees_df: pd.DataFrame, out_path: Path) -> Path:
    """Creates a line plot and writes a 300 DPI JPG."""
    x = degrees_df["star"].to_numpy()
    y = degrees_df["degree_of_connection"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.xlabel("7-Star Mapping Level")
    plt.ylabel("Degree of Connection (0–1)")
    plt.title("Degree of Connection for SKOS 7-Star Mapping")

    out_file = out_path / OUT_PLOT_JPG
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SKOS 7-star mapping RDF + degree plot from probly.csv"
    )
    parser.add_argument(
        "--csv",
        default="probly.csv",
        help="Input CSV (zonination/perceptions). Default: probly.csv",
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
        help="Output directory. Default: current directory",
    )

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    csv_path = (script_dir / args.csv).resolve()
    outdir = (script_dir / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_probly_csv(csv_path)
    med = phrase_medians(df)

    _, degrees_df = build_rdf(med, args.k, outdir)
    plot_file = plot_degrees(degrees_df, outdir)

    print("Done.")
    print(f"- Loaded rows: {len(df):,}")
    print(f"- Unique phrases: {len(med):,}")
    print(f"- RDF (Turtle): {outdir / OUT_TTL}")
    print(f"- Degrees CSV:  {outdir / OUT_DEGREES_CSV}")
    print(f"- Plot JPG:     {plot_file}")
    print()
    print("Degrees (star -> degree):")
    for _, row in degrees_df.iterrows():
        print(f"  {int(row['star'])} -> {row['degree_of_connection']:.6f}")


if __name__ == "__main__":
    main()
