"""
J-lens curves of ONE fact, one panel per linguistic type, both models (no averaging).

Each panel shows, for one prompt: δ with the assertion (thick; solid = the context won, dashed = memory
won) and δ0 for the same question without the assertion (thin dotted; one baseline prompt per fact,
so it is the same line on every panel). Uses only the saved analysis subsets and the data file
(no model).

Usage (from the repo root):
    python analysis/plot_one_fact.py --fact 338
Writes results/comparison/fact<N>_by_type.png (or --out DIR).
"""
import argparse
import json
from pathlib import Path
from textwrap import fill
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
ap.add_argument('--fact', type=int, default=338, help='fact_index (default: 338)')
ap.add_argument('--out', type=Path, default=ROOT / 'results/comparison', help='output directory')
args = ap.parse_args()
OUT, FACT = args.out, args.fact
OUT.mkdir(parents=True, exist_ok=True)
SUBSETS = {'qwen3-1.7b': 'results/qwen3-1.7b/qwen3-1.7b_4relations.parquet',
           'qwen3-4b': 'results/qwen3-4b/qwen3-4b_4relations.parquet'}
MCOL = {'qwen3-1.7b': '#6250d6', 'qwen3-4b': '#1baf7a'}
INK2, MUTED, GRID, BASE = '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
mpl.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'font.size': 8.5, 'axes.titlesize': 9, 'legend.frameon': False, 'savefig.bbox': 'tight'})
DIM_ORDER = ['evidentiality', 'epistemic_stance', 'form', 'tone']
data = pd.read_csv(ROOT / 'data/eobench_mcq_with_distractors.csv.gz')
fd = data[data.fact_index == FACT].set_index('category')
R = {}
for m, p in SUBSETS.items():
    df = pd.read_parquet(ROOT / p); R[m] = df[df.fact_index == FACT].set_index('category')
cats = (R['qwen3-1.7b'][['dimension']].reset_index().assign(k=lambda t: t.dimension.map(DIM_ORDER.index))
        .sort_values(['k', 'category']))
fig, axes = plt.subplots(5, 4, figsize=(17, 20), sharex=True, sharey=True)
for ax, (cat, dim) in zip(axes.flat, cats[['category', 'dimension']].itertuples(index=False)):
    sub = []
    for m, df in R.items():
        r = df.loc[cat]; x = json.loads(r.jlens_layers)
        won = r.choice == 'context'
        ax.plot(x, json.loads(r.delta_jlens), color=MCOL[m], lw=1.8, ls='-' if won else '--')
        ax.plot(x, json.loads(r.delta0_jlens), color=MCOL[m], lw=1.1, alpha=0.6, ls=(0, (1, 1.5)))
        sub.append(f"{m.split('-')[1].upper()}: {'CONTEXT won' if won else ('memory won' if r.choice == 'prior' else r.choice)}")
    ax.axhline(0, color=BASE, lw=1)
    ax.set_title(f'{cat} [{dim}]\n' + fill('«' + fd.loc[cat].assertion + '»', 60) + '\n' + '   '.join(sub), fontsize=8)
for ax in axes[-1]: ax.set_xlabel('layer')
for ax in axes[:, 0]: ax.set_ylabel('δ, J-lens (logits)')
leg = axes.flat[19]; leg.axis('off')
h = [Line2D([], [], color=MCOL[m], lw=2, label=m) for m in R] + [
     Line2D([], [], color=INK2, lw=1.8, label='δ WITH the assertion,\nthe context won'),
     Line2D([], [], color=INK2, lw=1.8, ls='--', label='δ WITH the assertion,\nmemory won'),
     Line2D([], [], color=INK2, lw=1.1, ls=(0, (1, 1.5)), label='δ0 WITHOUT the assertion\n(the same line on every panel)')]
leg.legend(handles=h, loc='center left', fontsize=9.5, labelspacing=1.1)
r0 = fd.iloc[0]
fig.suptitle(f'One fact (fact_index {FACT}): {r0.target_question}   context answer = {r0.context_answer}, '
             f'prior answer = {r0.prior_answer}, distractor = {r0.distractor_answer}.   J-lens δ per linguistic type',
             x=0.01, ha='left', fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.text(0.01, -0.003, fill('δ = logit(first token of the context answer) − logit(first token of the prior answer), one prompt per curve (no averaging). '
         'δ0 = the same question without the assertion: one baseline prompt per fact, so the thin line is identical on all 19 panels. '
         'J-lens has no matrix for the last layer; x = absolute layer number.', 230), ha='left', va='top', fontsize=8.5, color=INK2)
fig.savefig(OUT / f'fact{FACT}_by_type.png', dpi=120)
print('saved', OUT / f'fact{FACT}_by_type.png')
