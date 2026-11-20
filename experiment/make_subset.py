"""
Builds the analysis subset from the raw output of run_flip.py.

The full qwen3-1.7b run was stopped after 11,514 of 65,778 rows. The dataset is sorted
by fact and by relation, so the rows that were done cover facts 0-610, i.e. only
the relations author, capital, color and part of composer. This script:

  1. reads the raw results (one .parquet file, or a *_parts/ folder of run_flip.py);
  2. keeps only facts with all 19 linguistic variants (drops the last, partly done fact);
  3. adds the `relation` column from the dataset;
  4. sorts by fact, dimension, category and saves .parquet (for analysis) and .csv (to look at).

Usage (from the repo root):
    python experiment/make_subset.py \
        --raw results/qwen3-1.7b/raw/qwen3-1.7b_raw.parquet \
        --data data/eobench_mcq_with_distractors.csv.gz \
        --out results/qwen3-1.7b/qwen3-1.7b_4relations.parquet
"""
import argparse
from pathlib import Path

import pandas as pd

N_VARIANTS = 19  # 4 dimensions: evidentiality 3 + epistemic stance 2 + form 7 + tone 7


def read_raw(path):
    """One parquet file, or a folder with part_*.parquet files written by run_flip.py."""
    p = Path(path)
    if p.is_dir():
        parts = sorted(p.glob("part_*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    return pd.read_parquet(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw results: .parquet file or *_parts/ folder")
    ap.add_argument("--data", required=True, help="dataset (.csv or .csv.gz) to take `relation` from")
    ap.add_argument("--out", required=True, help="output .parquet; a .csv copy is written next to it")
    args = ap.parse_args()

    df = read_raw(args.raw).drop_duplicates("id")
    n_per_fact = df.groupby("fact_index").id.transform("size")
    dropped = sorted(int(f) for f in df.loc[n_per_fact != N_VARIANTS, "fact_index"].unique())
    df = df[n_per_fact == N_VARIANTS]

    rel = pd.read_csv(args.data, usecols=["id", "relation"])
    df = df.merge(rel, on="id", how="left")
    df.insert(2, "relation", df.pop("relation"))
    df = df.sort_values(["fact_index", "dimension", "category"]).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)

    print(f"dropped incomplete facts: {dropped}")
    print(f"saved {len(df)} rows, {df.fact_index.nunique()} facts -> {out} (+ .csv)")
    print(df.drop_duplicates("fact_index").relation.value_counts().to_string())


if __name__ == "__main__":
    main()
