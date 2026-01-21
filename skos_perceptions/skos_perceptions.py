#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKOS-Perceptions Mapping (SKOS-Plus)
===================================

This script models empirically observed probability phrases (from zonination/perceptions)
as **SKOS-compatible mapping properties** with numeric degrees of connection.

Inputs
------
- probly.csv : Perceptions-of-probability dataset (wide format preferred; long format supported)

Outputs
-------
- skos_perceptions_mapping.ttl        : RDF/Turtle (property-only TBox)
- skos_perceptions_stats.csv          : Median/Q1/Q3/IQR/whiskers + SKOS parent assignment
- skos_perceptions_degree_plot.jpg    : 300 DPI plot (17 medians + logistic approximation curve)

Modelling notes
--------------
- Property-only RDF (TBox): **no individuals** are created.
- Phrases are modelled as owl:ObjectProperty and placed under exactly one SKOS mapping property:
    * skos:exactMatch   if median in [0.9382 .. 1.0]
    * skos:closeMatch   if median in [0.4948 .. 0.9381]
    * skos:relatedMatch if median in [0.0 .. 0.4948)
- Numeric justification:
    * skosplus:degreeOfConnection (0..1) equals the empirical median (median/100)
    * additional annotations: Q1, Q3, IQR, Tukey whiskers

Namespace
---------
- skosplus: https://w3id.org/skos-plus/

Usage
-----
Run from within the folder (where probly.csv is located):

    python skos_perceptions.py

Optional arguments:

    python skos_perceptions.py --help
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD


# -----------------------------
# CONFIG
# -----------------------------

SKOSPLUS_NS = "https://w3id.org/skos-plus/"
PERCEPTIONS_REPO = "https://github.com/zonination/perceptions"

OUT_TTL = "skos_perceptions_mapping.ttl"
OUT_STATS_CSV = "skos_perceptions_stats.csv"
OUT_PLOT_JPG = "skos_perceptions_degree_plot.jpg"

# Thresholds (given / agreed)
EXACT_MIN = 0.9382
CLOSE_MIN = 0.4948

# Logistic approximation defaults (used for smooth curve only; medians remain the reference)
DEFAULT_LOGISTIC_K = 0.348
DEFAULT_LOGISTIC_R0 = 9.73

EXPORT_DPI = 300


# -----------------------------
# TEXT / IDs
# -----------------------------


def slug_phrase(phrase: str) -> str:
    """Stable localname suffix for phrase properties."""
    return (
        phrase.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
    )


# -----------------------------
# IO
# -----------------------------


def load_probly_wide_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load perceptions CSV in *wide* format:
    - columns are phrases
    - rows are numeric values (typically 0..100)

    If your CSV is already long-format (phrase/probability), you can adapt it before
    calling this script; however, our repository baseline uses wide format.
    """
    df = pd.read_csv(csv_path)
    # Coerce all columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# -----------------------------
# STATS
# -----------------------------


def to_unit_interval(series: pd.Series) -> pd.Series:
    """Convert 0..100 percentages to 0..1 if needed and filter to [0,1]."""
    s = series.dropna().astype(float)
    if len(s) == 0:
        return s
    if s.max() > 1.0:
        s = s / 100.0
    return s[(s >= 0.0) & (s <= 1.0)]


def tukey_whiskers(values: pd.Series, q1: float, q3: float) -> Tuple[float, float]:
    """
    Tukey whiskers:
    - lower = max(observed_min, Q1 - 1.5*IQR)
    - upper = min(observed_max, Q3 + 1.5*IQR)
    """
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    vmin = float(values.min())
    vmax = float(values.max())
    lower = max(vmin, lower_fence)
    upper = min(vmax, upper_fence)
    return float(lower), float(upper)


# -----------------------------
# MAPPING / CURVES
# -----------------------------


def classify_skos_parent(median: float):
    """Assign the phrase property to exactly one SKOS mapping property based on the median."""
    if median >= EXACT_MIN:
        return SKOS.exactMatch
    if median >= CLOSE_MIN:
        return SKOS.closeMatch
    return SKOS.relatedMatch


def logistic(r: float, k: float, r0: float) -> float:
    """Logistic approximation curve (0..1)."""
    return 1.0 / (1.0 + math.exp(-k * (r - r0)))


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


def build_rdf_and_stats(df_wide: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """
    Build RDF (Turtle) and return a stats table (DataFrame).
    Writes:
    - OUT_TTL
    - OUT_STATS_CSV
    """
    g = Graph()
    skosplus = bind_common_prefixes(g)
    ensure_skos_backbone(g)

    # Core annotation properties (keep stable across scripts)
    DEG = skosplus.degreeOfConnection
    MED = skosplus.medianPerceivedProbability

    # Distribution stats (perceptions layer)
    Q1 = skosplus.q1PerceivedProbability
    Q3 = skosplus.q3PerceivedProbability
    IQR = skosplus.iqrPerceivedProbability
    WL = skosplus.lowerWhiskerPerceivedProbability
    WU = skosplus.upperWhiskerPerceivedProbability

    declare_annotation_property(
        g,
        DEG,
        "degree of connection (0–1)",
        "Numeric degree; for perceptions this equals the empirical median.",
    )
    declare_annotation_property(
        g,
        MED,
        "median perceived probability",
        "Empirical median (0..1) derived from zonination/perceptions.",
    )
    declare_annotation_property(
        g, Q1, "Q1 perceived probability", "First quartile (0..1)."
    )
    declare_annotation_property(
        g, Q3, "Q3 perceived probability", "Third quartile (0..1)."
    )
    declare_annotation_property(
        g, IQR, "IQR perceived probability", "Interquartile range (Q3-Q1)."
    )
    declare_annotation_property(
        g,
        WL,
        "lower whisker perceived probability",
        "Lower Tukey whisker (clipped to observed min).",
    )
    declare_annotation_property(
        g,
        WU,
        "upper whisker perceived probability",
        "Upper Tukey whisker (clipped to observed max).",
    )

    rows = []
    for phrase in df_wide.columns:
        vals = to_unit_interval(df_wide[phrase])
        if len(vals) == 0:
            continue

        q1 = float(vals.quantile(0.25))
        med = float(vals.median())
        q3 = float(vals.quantile(0.75))
        iqr = float(q3 - q1)
        wlow, whigh = tukey_whiskers(vals, q1, q3)

        skos_parent = classify_skos_parent(med)

        local = f"perceptions_{slug_phrase(phrase)}"
        p_uri = skosplus[local]

        # Phrase property as ObjectProperty (property-only model)
        g.add((p_uri, RDF.type, OWL.ObjectProperty))
        g.add((p_uri, RDFS.label, Literal(str(phrase), lang="en")))
        g.add((p_uri, DCTERMS.source, Literal(PERCEPTIONS_REPO)))

        # Key hierarchy: phrase property is placed under exactly one SKOS mapping property
        g.add((p_uri, RDFS.subPropertyOf, skos_parent))

        # Domain/Range: Concept-to-Concept mapping context
        g.add((p_uri, RDFS.domain, SKOS.Concept))
        g.add((p_uri, RDFS.range, SKOS.Concept))

        # Numeric justification annotations
        g.add((p_uri, DEG, Literal(med, datatype=XSD.decimal)))
        g.add((p_uri, MED, Literal(med, datatype=XSD.decimal)))
        g.add((p_uri, Q1, Literal(q1, datatype=XSD.decimal)))
        g.add((p_uri, Q3, Literal(q3, datatype=XSD.decimal)))
        g.add((p_uri, IQR, Literal(iqr, datatype=XSD.decimal)))
        g.add((p_uri, WL, Literal(wlow, datatype=XSD.decimal)))
        g.add((p_uri, WU, Literal(whigh, datatype=XSD.decimal)))

        rows.append(
            {
                "phrase": str(phrase),
                "slug": local,
                "degree_of_connection": med,
                "median": med,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_whisker": wlow,
                "upper_whisker": whigh,
                "skos_parent": str(skos_parent),
                "property_uri": str(p_uri),
            }
        )

    stats = (
        pd.DataFrame(rows).sort_values("degree_of_connection").reset_index(drop=True)
    )

    outdir.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(outdir / OUT_TTL), format="turtle")
    stats.to_csv(outdir / OUT_STATS_CSV, index=False, encoding="utf-8")

    return stats


# -----------------------------
# PLOT
# -----------------------------


def plot_stats(stats: pd.DataFrame, outdir: Path, k: float, r0: float) -> Path:
    """
    Plot the 17 median points (ordered by median) plus a logistic approximation curve.
    Writes: OUT_PLOT_JPG
    """
    stats_sorted = stats.sort_values("median").reset_index(drop=True)

    y = stats_sorted["median"].to_numpy(dtype=float)
    x = stats_sorted.index.to_numpy(dtype=float) + 1.0  # 1..N

    x_curve = pd.Series([1.0 + i * ((len(x) - 1.0) / 500.0) for i in range(501)])
    y_curve = x_curve.apply(lambda rr: logistic(float(rr), k, r0)).to_numpy(dtype=float)

    plt.figure(figsize=(12, 6))
    plt.plot(x_curve, y_curve)  # approximation curve
    plt.plot(x, y, marker="o", linestyle="")  # empirical medians

    plt.ylim(0, 1.05)
    plt.xlim(1, len(x))
    plt.xticks(range(1, len(x) + 1))
    plt.xlabel("Phrase rank r (ordered by median)")
    plt.ylabel("Degree of Connection (median / 100)")
    plt.title(
        "Degree of Connection for Perceptions Phrases (17 medians + logistic curve)"
    )

    out_file = outdir / OUT_PLOT_JPG
    plt.savefig(out_file, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close()
    return out_file


# -----------------------------
# MAIN
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SKOS-Perceptions RDF (properties only) + plot from probly.csv"
    )
    parser.add_argument(
        "--csv",
        default="probly.csv",
        help="Input CSV (wide format, columns=phrases). Default: probly.csv",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory (relative to this script). Default: current script folder",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_LOGISTIC_K,
        help=f"Logistic curve parameter k. Default: {DEFAULT_LOGISTIC_K}",
    )
    parser.add_argument(
        "--r0",
        type=float,
        default=DEFAULT_LOGISTIC_R0,
        help=f"Logistic curve parameter r0. Default: {DEFAULT_LOGISTIC_R0}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    csv_path = (script_dir / args.csv).resolve()
    outdir = (script_dir / args.outdir).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df_wide = load_probly_wide_csv(csv_path)
    stats = build_rdf_and_stats(df_wide, outdir)
    plot_file = plot_stats(stats, outdir, args.k, args.r0)

    # Consistent summary output
    print("Done.")
    print(f"- CSV file:      {csv_path}")
    print(f"- Phrases cols:  {df_wide.shape[1]}")
    print(f"- RDF (Turtle):  {outdir / OUT_TTL}")
    print(f"- Stats CSV:     {outdir / OUT_STATS_CSV}")
    print(f"- Plot JPG:      {plot_file}")
    print()
    print("Mapping overview (phrase -> median -> SKOS parent):")
    for _, row in stats.sort_values("median", ascending=True).iterrows():
        print(f"  {row['phrase']}: {row['median']:.4f} -> {row['skos_parent']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
