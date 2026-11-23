"""
Robustness checks for the Qwen3-1.7B vs Qwen3-4B comparison (appendix of results/comparison/findings.md).

Uses only the saved analysis subsets (no model). Three checks:
  1. 95% bootstrap confidence intervals for % context won per linguistic type, per model.
     Facts are resampled (not rows), because the 19 variants of one fact are not independent.
  2. The comparison on facts that BOTH models know. Each model's own numbers use only the facts
     that model knows, and 4B knows more facts, so the two sets differ.
  3. Correlation across the 19 types between % context won and the mean divergence layer,
     computed on all rows and separately within context-won and within prior-won rows.
     On all rows part of the correlation is mechanical: context-won rows diverge earlier, so a type
     with more context-won rows gets an earlier mean.

Usage (from the repo root):
    python analysis/robustness.py
Writes results/comparison/robustness/*.csv and prints a summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = {"qwen3-1.7b": ROOT / "results/qwen3-1.7b/qwen3-1.7b_4relations.parquet",
           "qwen3-4b": ROOT / "results/qwen3-4b/qwen3-4b_4relations.parquet"}
OUT = ROOT / "results/comparison/robustness"
N_BOOT = 2000
SEED = 0


def usable(df, facts=None):
    """Rows used in the analysis: known facts, final answer = context or prior."""
    ok = df[df.knows_prior & df.choice.isin(["context", "prior"])]
    return ok if facts is None else ok[ok.fact_index.isin(facts)]


def counts(ok, facts, cats):
    """Arrays [n_facts, n_types] of context-won rows and of all rows."""
    t = ok.assign(ctx=ok.choice.eq("context")).pivot_table(
        index="fact_index", columns="category", values="ctx", aggfunc=["sum", "count"], fill_value=0)
    won = t["sum"].reindex(index=facts, columns=cats, fill_value=0).to_numpy(float)
    n = t["count"].reindex(index=facts, columns=cats, fill_value=0).to_numpy(float)
    return won, n


def boot_pct(won, n, rng):
    """Point % and bootstrap 95% CI per type (and overall in the last column), resampling facts."""
    idx = rng.integers(0, len(won), size=(N_BOOT, len(won)))
    w, m = won[idx].sum(1), n[idx].sum(1)                      # [N_BOOT, n_types]
    per = 100 * w / m
    tot = 100 * w.sum(1) / m.sum(1)
    point = np.append(100 * won.sum(0) / n.sum(0), 100 * won.sum() / n.sum())
    samples = np.column_stack([per, tot])
    return point, samples


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data = {m: pd.read_parquet(p) for m, p in SUBSETS.items()}
    cats = sorted(data["qwen3-1.7b"].category.unique())
    dims = data["qwen3-1.7b"].drop_duplicates("category").set_index("category").dimension

    # 1. per-model CIs (each model on the facts it knows: the numbers of the main findings)
    rows = []
    for m, df in data.items():
        ok = usable(df)
        facts = sorted(ok.fact_index.unique())
        point, s = boot_pct(*counts(ok, facts, cats), rng)
        lo, hi = np.percentile(s, [2.5, 97.5], axis=0)
        for i, c in enumerate(cats + ["ALL"]):
            rows.append({"model": m, "category": c, "facts": len(facts), "context_won_%": point[i],
                         "ci_low": lo[i], "ci_high": hi[i]})
    own = pd.DataFrame(rows).round(1)
    own.to_csv(OUT / "context_won_ci_own_facts.csv", index=False)

    # 2. facts known by both models: paired bootstrap of the 4B - 1.7B difference
    known = {m: set(df[df.knows_prior].fact_index) for m, df in data.items()}
    shared = sorted(known["qwen3-1.7b"] & known["qwen3-4b"])
    idx = rng.integers(0, len(shared), size=(N_BOOT, len(shared)))
    res = {}
    for m, df in data.items():
        won, n = counts(usable(df, shared), shared, cats)
        point = np.append(100 * won.sum(0) / n.sum(0), 100 * won.sum() / n.sum())
        w, k = won[idx].sum(1), n[idx].sum(1)
        res[m] = (point, np.column_stack([100 * w / k, 100 * w.sum(1) / k.sum(1)]))
    diff_pt = res["qwen3-4b"][0] - res["qwen3-1.7b"][0]
    diff_s = res["qwen3-4b"][1] - res["qwen3-1.7b"][1]
    lo, hi = np.percentile(diff_s, [2.5, 97.5], axis=0)
    shared_tab = pd.DataFrame({
        "dimension": [dims.get(c, "") for c in cats] + [""], "category": cats + ["ALL"],
        "qwen3-1.7b_%": res["qwen3-1.7b"][0], "qwen3-4b_%": res["qwen3-4b"][0],
        "diff_4b_minus_1.7b": diff_pt, "diff_ci_low": lo, "diff_ci_high": hi,
    }).round(1)
    shared_tab["ci_excludes_0"] = (shared_tab.diff_ci_low > 0) | (shared_tab.diff_ci_high < 0)
    shared_tab.to_csv(OUT / "comparison_shared_facts.csv", index=False)
    per_type = shared_tab[shared_tab.category != "ALL"]
    rho = per_type["qwen3-1.7b_%"].rank().corr(per_type["qwen3-4b_%"].rank())

    # 3. correlation across types: % context won vs mean divergence layer (J-lens)
    corr_rows = []
    for m, df in data.items():
        ok = usable(df)
        t = ok.groupby("category").agg(ctx=("choice", lambda s: (s == "context").mean()),
                                       div_all=("divergence_jl", "mean"))
        t["div_ctx"] = ok[ok.choice == "context"].groupby("category").divergence_jl.mean()
        t["div_pri"] = ok[ok.choice == "prior"].groupby("category").divergence_jl.mean()
        corr_rows.append({"model": m,
                          "r_all_rows": t.ctx.corr(t.div_all),
                          "r_context_won_rows": t.ctx.corr(t.div_ctx),
                          "r_prior_won_rows": t.ctx.corr(t.div_pri)})
    corr = pd.DataFrame(corr_rows).round(2)
    corr.to_csv(OUT / "correlation_within_groups.csv", index=False)

    # summary
    print(f"bootstrap: {N_BOOT} resamples of facts, seed {SEED}\n")
    print("1. % context won with 95% CI, each model on the facts it knows")
    print(own.pivot(index="category", columns="model",
                    values=["context_won_%", "ci_low", "ci_high"]).to_string(), "\n")
    print(f"2. facts known by both models: {len(shared)} "
          f"(1.7B knows {len(known['qwen3-1.7b'])}, 4B knows {len(known['qwen3-4b'])})")
    print(shared_tab.sort_values("diff_4b_minus_1.7b").to_string(index=False))
    print(f"   Spearman rho of the type ranking on shared facts: {rho:.2f}\n")
    print("3. correlation across the 19 types: % context won vs mean divergence layer (J-lens)")
    print(corr.to_string(index=False))
    print(f"\nsaved {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
