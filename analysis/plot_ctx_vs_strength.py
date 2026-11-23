"""
% context won as a function of the strength of the fact, one line per linguistic type, both models.

Strength of a fact = δ0 = logit(context answer) - logit(prior answer) for the question WITHOUT the
assertion, at the last layer (logit lens). Rows (known facts, final answer = context or prior) are
split into 5 equal-size groups by δ0. If only the strength of the fact mattered, the 19 lines would
lie on top of each other; the vertical spread at the same strength is the effect of the wording.
Uses only the saved analysis subsets (no model).

Usage (from the repo root):
    python analysis/plot_ctx_vs_strength.py
Writes results/comparison/ctx_vs_strength.png (or --out DIR).
"""
import argparse
import json
from pathlib import Path
from textwrap import fill
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
ap.add_argument('--out', type=Path, default=ROOT / 'results/comparison', help='output directory')
ap.add_argument('--bins', type=int, default=5, help='number of equal-size strength groups (default: 5)')
args = ap.parse_args()
OUT, NB = args.out, args.bins
OUT.mkdir(parents=True, exist_ok=True)
INK2, MUTED, GRID, BASE = '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
mpl.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'font.size': 9, 'axes.titlesize': 10.5, 'legend.frameon': False, 'savefig.bbox': 'tight'})
HL = {'authority': '#2a78d6', 'imperative': '#1baf7a', 'belief_reports': '#e34948', 'weak': '#d18a00'}
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for ax, m in zip(axes, ['qwen3-1.7b', 'qwen3-4b']):
    r = pd.read_parquet(ROOT / f'results/{m}/{m}_4relations.parquet')
    d = r[r.knows_prior & r.choice.isin(['context', 'prior'])].copy()
    d['ctx'] = d.choice.eq('context'); d['d0'] = d.delta0_logit_lens.map(lambda v: json.loads(v)[-1])
    d['q'] = pd.qcut(d.d0, NB, labels=False)
    xm = d.groupby('q').d0.mean()                      # x = mean delta0 of each strength group
    t = d.pivot_table(index='category', columns='q', values='ctx', aggfunc='mean') * 100
    for cat, row in t.iterrows():
        if cat not in HL:
            ax.plot(xm, row.values, color='#b9b8b0', lw=1, alpha=0.8, zorder=1)
    for cat, col in HL.items():
        ax.plot(xm, t.loc[cat].values, color=col, lw=2.2, marker='o', ms=6, zorder=3,
                markeredgecolor='#fcfcfb', markeredgewidth=1.5)
        ax.annotate(cat, (xm.iloc[0], t.loc[cat].iloc[0]), xytext=(-8, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=9, color=INK2)
    allc = d.groupby('q').ctx.mean() * 100
    ax.plot(xm, allc.values, color='#0b0b0b', lw=2.6, ls='--', zorder=4)
    spread = (t.max() - t.min()).round(0).astype(int).tolist()
    ax.set_xlim(xm.min() - 9, xm.max() + 3); ax.set_ylim(-2, 104)
    ax.set_xticks(xm.values, [f'Q{i + 1}\n{v:.0f}\nrange {sp} pt' for i, (v, sp) in enumerate(zip(xm, spread))])
    ax.set_xlabel('strength of memory: δ0 quintile (Q1 = strongest), mean δ0 in logits')
    ax.set_title(f'{m}: % context won vs strength of the fact ({len(d):,} rows, {d.fact_index.nunique()} facts)', loc='left')
axes[0].set_ylabel('% of rows where the context answer won')
from matplotlib.lines import Line2D
axes[1].legend(handles=[Line2D([], [], color=c, lw=2.2, marker='o', label=k) for k, c in HL.items()] +
               [Line2D([], [], color='#b9b8b0', lw=1, label='other 15 types'),
                Line2D([], [], color='#0b0b0b', lw=2.6, ls='--', label='all 19 types pooled')], loc='upper left', bbox_to_anchor=(0.0, 1.0))
fig.tight_layout()
txt = ['Strength of a fact = δ0 = logit(context answer) − logit(prior answer) for the question WITHOUT the assertion, last layer, logit lens '
       '(more negative = the model is more sure of its memorised answer). Rows are split into 5 equal-size groups by δ0; x = mean δ0 of the group. '
       'One line per linguistic type; "range" = max − min of % context won across the 19 types in that group. '
       'Only facts the model knows; distractor rows left out. The same facts are used for all 19 types.']
fig.text(0.01, -0.01, '\n'.join(fill(t, 210) for t in txt), ha='left', va='top', fontsize=8.5, color=INK2, linespacing=1.4)
fig.savefig(OUT / 'ctx_vs_strength.png', dpi=140)
print('saved', OUT / 'ctx_vs_strength.png')
