#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SKOS-Perceptions mapping (SKOS-Plus)
-----------------------------------
Goal:
- Model the 17 probability phrases from zonination/perceptions as *properties* (no individuals).
- Each phrase property gets a numeric degreeOfConnection derived from the *empirical median* (median/100).
- Each phrase property becomes a subPropertyOf exactly ONE of:
    - skos:exactMatch   if median in [0.9382 .. 1.0]
    - skos:closeMatch   if median in [0.4948 .. 0.9381]
    - skos:relatedMatch if median in [0.0 .. 0.4948)
- Additionally, store distribution stats needed to justify the degree:
    - Q1, Median, Q3, IQR, lower/upper whisker (Tukey 1.5*IQR)
- Plot:
    - x-axis: phrase rank r = 1..17 (ordered by median)
    - y-axis: degreeOfConnection (median/100)
    - overlay: logistic approximation curve (simple, reproducible)

Folder assumptions (VS Code / GitHub repo):
- This script file:   skos_perceptions/skos_perceptions.py
- CSV file:           skos_perceptions/probly.csv

Outputs (written next to this script by default):
- skos_perceptions_mapping.ttl
- skos_perceptions_stats.csv
- skos_perceptions_degree_plot.jpg  (300 DPI)

Source:
- https://github.com/zonination/perceptions
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS, DCTERMS


# -----------------------------
# CONFIG
# -----------------------------
SKOSPLUS_NS = "https://w3id.org/skos-plus/"
PERCEPTIONS_REPO = "https://github.com/zonination/perceptions"

OUT_TTL = "skos_perceptions_mapping.ttl"
OUT_STATS_CSV = "skos_perceptions_stats.csv"
OUT_PLOT_JPG = "skos_perceptions_degree_plot.jpg"

# Your SKOS mapping band thresholds (given)
EXACT_MIN = 0.9382
CLOSE_MIN = 0.4948

# Logistic approximation defaults (fit once from the 17 medians; used as *approximation*)
DEFAULT_LOGISTIC_K = 0.348
DEFAULT_LOGISTIC_R0 = 9.73


# -----------------------------
# Helpers
# -----------------------------
def _slug_phrase(phrase: str) -> str:
    """Stable localname suffix for phrase properties."""
    return (
        phrase.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
    )


def load_probly_wide_csv(csv_path: Path) -> pd.DataFrame:
    """
    Loads the perceptions CSV in wide format:
    - columns are phrases
    - rows are numeric values (usually 0..100)
    Returns a numeric-only dataframe.
    """
    df = pd.read_csv(csv_path)
    # coerce all columns to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def to_unit_interval(series: pd.Series) -> pd.Series:
    """Convert 0..100 percentages to 0..1 if needed."""
    s = series.dropna().astype(float)
    if len(s) == 0:
        return s
    if s.max() > 1.0:
        s = s / 100.0
    return s[(s >= 0.0) & (s <= 1.0)]


def tukey_whiskers(values: pd.Series, q1: float, q3: float) -> Tuple[float, float]:
    """
    Tukey whiskers: lower=max(min, Q1-1.5*IQR), upper=min(max, Q3+1.5*IQR)
    using observed data bounds.
    """
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    vmin = float(values.min())
    vmax = float(values.max())
    lower = max(vmin, lower_fence)
    upper = min(vmax, upper_fence)
    return float(lower), float(upper)


def classify_skos_parent(median: float):
    """
    Map median degree to SKOS property, based on the user-defined ranges.
    """
    if median >= EXACT_MIN:
        return SKOS.exactMatch
    if median >= CLOSE_MIN:
        return SKOS.closeMatch
    return SKOS.relatedMatch


def logistic(r: float, k: float, r0: float) -> float:
    """Logistic approximation curve (0..1)."""
    return 1.0 / (1.0 + math.exp(-k * (r - r0)))


# -----------------------------
# RDF Building
# -----------------------------
def build_rdf_and_stats(
    df_wide: pd.DataFrame,
    outdir: Path,
) -> pd.DataFrame:
    """
    Returns a stats dataframe and writes RDF (Turtle).
    """
    SKOSPLUS = Namespace(SKOSPLUS_NS)

    g = Graph()
    g.bind("skos", SKOS)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    g.bind("dct", DCTERMS)
    g.bind("skosplus", SKOSPLUS)

    # TBox: explicit class (no individuals)
    g.add((SKOS.Concept, RDF.type, OWL.Class))

    # Ensure SKOS match properties are object properties (for Protégé views)
    for p in (SKOS.relatedMatch, SKOS.closeMatch, SKOS.exactMatch):
        g.add((p, RDF.type, OWL.ObjectProperty))
        g.add((p, RDFS.domain, SKOS.Concept))
        g.add((p, RDFS.range, SKOS.Concept))

    # Annotation properties (keep consistent with earlier scripts)
    DEG = SKOSPLUS.degreeOfConnection
    MED = SKOSPLUS.medianPerceivedProbability

    # Additional distribution annotations (new, but still under skosplus:)
    Q1 = SKOSPLUS.q1PerceivedProbability
    Q3 = SKOSPLUS.q3PerceivedProbability
    IQR = SKOSPLUS.iqrPerceivedProbability
    WL = SKOSPLUS.lowerWhiskerPerceivedProbability
    WU = SKOSPLUS.upperWhiskerPerceivedProbability

    for ap, label, comment in [
        (
            DEG,
            "degree of connection (0–1)",
            "Numeric degree; for perceptions this equals the empirical median.",
        ),
        (
            MED,
            "median perceived probability",
            "Empirical median (0..1) from zonination/perceptions.",
        ),
        (
            Q1,
            "Q1 perceived probability",
            "First quartile (0..1) from zonination/perceptions.",
        ),
        (
            Q3,
            "Q3 perceived probability",
            "Third quartile (0..1) from zonination/perceptions.",
        ),
        (IQR, "IQR perceived probability", "Interquartile range (Q3-Q1)."),
        (
            WL,
            "lower whisker perceived probability",
            "Lower Tukey whisker (Q1-1.5*IQR, clipped to observed min).",
        ),
        (
            WU,
            "upper whisker perceived probability",
            "Upper Tukey whisker (Q3+1.5*IQR, clipped to observed max).",
        ),
    ]:
        g.add((ap, RDF.type, OWL.AnnotationProperty))
        g.add((ap, RDFS.label, Literal(label, lang="en")))
        g.add((ap, RDFS.comment, Literal(comment, lang="en")))

    # Build stats table
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

        local = f"perceptions_{_slug_phrase(phrase)}"
        p_uri = SKOSPLUS[local]

        # Phrase property as ObjectProperty (property-only model)
        g.add((p_uri, RDF.type, OWL.ObjectProperty))
        g.add((p_uri, RDFS.label, Literal(str(phrase), lang="en")))
        g.add((p_uri, DCTERMS.source, Literal(PERCEPTIONS_REPO)))

        # The key hierarchy: phrase is subPropertyOf the SKOS mapping property
        g.add((p_uri, RDFS.subPropertyOf, skos_parent))

        # Domain/Range: same as mapping props (Concept-to-Concept mapping context)
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

    # write outputs
    outdir.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(outdir / OUT_TTL), format="turtle")
    stats.to_csv(outdir / OUT_STATS_CSV, index=False, encoding="utf-8")

    return stats


# -----------------------------
# Plot
# -----------------------------
def plot_stats(stats: pd.DataFrame, outdir: Path, k: float, r0: float) -> Path:
    """
    Plot: 17 median points + logistic approximation curve.
    x-axis uses rank r = 1..N ordered by median.
    """
    stats_sorted = stats.sort_values("median").reset_index(drop=True)
    y = stats_sorted["median"].to_numpy(dtype=float)
    x = stats_sorted.index.to_numpy(dtype=float) + 1.0  # 1..17

    x_curve = [1.0 + i * (16.0 / 500.0) for i in range(501)]  # 1..17
    y_curve = [logistic(rr, k, r0) for rr in x_curve]

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
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


# -----------------------------
# Main
# -----------------------------
def main() -> None:
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
        help="Output directory (relative to script dir). Default: current script folder",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_LOGISTIC_K,
        help=f"Logistic approximation parameter k. Default: {DEFAULT_LOGISTIC_K}",
    )
    parser.add_argument(
        "--r0",
        type=float,
        default=DEFAULT_LOGISTIC_R0,
        help=f"Logistic approximation parameter r0. Default: {DEFAULT_LOGISTIC_R0}",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    csv_path = (script_dir / args.csv).resolve()
    outdir = (script_dir / args.outdir).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df_wide = load_probly_wide_csv(csv_path)
    stats = build_rdf_and_stats(df_wide, outdir)
    plot_file = plot_stats(stats, outdir, args.k, args.r0)

    print("Done.")
    print(f"- CSV loaded columns (phrases): {df_wide.shape[1]}")
    print(f"- RDF (Turtle): {outdir / OUT_TTL}")
    print(f"- Stats CSV:    {outdir / OUT_STATS_CSV}")
    print(f"- Plot JPG:     {plot_file}")
    print()
    print("Mapping overview (phrase -> median -> skos parent):")
    for _, row in stats.sort_values("median", ascending=True).iterrows():
        print(f"  {row['phrase']}: {row['median']:.4f} -> {row['skos_parent']}")


if __name__ == "__main__":
    main()
