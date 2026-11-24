"""
cmp3 with the baseline added: J-lens δ on absolute layers, both models, with δ0 (the same question
WITHOUT the assertion) drawn as thin lines.

Left: mean δ with the assertion (thick) and mean δ0 (thin). Right: mean shift δ - δ0 (thick) and
mean δ0 (thin). Rows are known facts, split by the final answer WITH the assertion (context won /
memory won); the thin lines are the baseline of the same rows. Uses only the saved analysis subsets
(no model).

Usage (from the repo root):
    python analysis/plot_cmp3_delta0.py
Writes results/comparison/cmp3_with_delta0.png (or --out DIR).
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
ap.add_argument('--out', type=Path, default=ROOT / 'results/comparison', help='output directory')
args = ap.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
SUBSETS = {'qwen3-1.7b': 'results/qwen3-1.7b/qwen3-1.7b_4relations.parquet',
           'qwen3-4b': 'results/qwen3-4b/qwen3-4b_4relations.parquet'}
MCOL = {'qwen3-1.7b': '#6250d6', 'qwen3-4b': '#1baf7a'}
INK2, MUTED, GRID, BASE = '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
mpl.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2, 'font.size': 9, 'axes.titlesize': 10,
    'legend.frameon': False, 'savefig.bbox': 'tight'})
D = {}
for m, p in SUBSETS.items():
    raw = pd.read_parquet(ROOT / p)
    df = raw[raw.knows_prior & raw.choice.isin(['context', 'prior'])].reset_index(drop=True)
    st = lambda c: np.stack(df[c].map(json.loads))
    LL, JL, LL0, JL0 = st('delta_logit_lens'), st('delta_jlens'), st('delta0_logit_lens'), st('delta0_jlens')
    nl = LL.shape[1]; is_ctx = (df.choice == 'context').values
    gap = LL[is_ctx].mean(0) - LL[~is_ctx].mean(0); sep = int(np.where(gap > 2)[0][0])
    xj = np.array(json.loads(df.jlens_layers.iloc[0]))
    D[m] = dict(S_LL=LL - LL0, S_JL=JL - JL0, D0_LL=LL0, D0_JL=JL0, r_ll=np.arange(nl) / (nl - 1),
                r_jl=xj / (nl - 1), x_jl=xj, JL=JL, is_ctx=is_ctx, sep=sep, nl=nl)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
for ax, key, ttl, yl in [(axes[0], 'JL', 'J-lens: mean δ with the assertion (thick) and δ0 without it (thin)', 'mean δ (logits)'),
                         (axes[1], 'S_JL', 'J-lens: shift of δ caused by the assertion (thick) and δ0 (thin)', 'logits')]:
    for m, d in D.items():
        c = d['is_ctx']
        ax.plot(d['x_jl'], d[key][c].mean(0), color=MCOL[m], lw=2)
        ax.plot(d['x_jl'], d[key][~c].mean(0), color=MCOL[m], lw=2, ls='--')
        ax.plot(d['x_jl'], d['D0_JL'][c].mean(0), color=MCOL[m], lw=1.3, alpha=0.6, ls='-.')
        ax.plot(d['x_jl'], d['D0_JL'][~c].mean(0), color=MCOL[m], lw=1.3, alpha=0.6, ls=(0, (1, 1.5)))
    ax.axhline(0, color=BASE, lw=1); ax.set_title(ttl, loc='left'); ax.set_xlabel('layer'); ax.set_ylabel(yl)
h = [Line2D([], [], color=MCOL[m], lw=2, label=m) for m in D] + [
     Line2D([], [], color=INK2, lw=2, label='with the assertion, context won'),
     Line2D([], [], color=INK2, lw=2, ls='--', label='with the assertion, prior won'),
     Line2D([], [], color=INK2, lw=1.3, ls='-.', label='δ0 (no assertion), context-won rows'),
     Line2D([], [], color=INK2, lw=1.3, ls=(0, (1, 1.5)), label='δ0 (no assertion), prior-won rows')]
axes[0].legend(handles=h, loc='upper left')
fig.tight_layout()
txt = ['Left: raw δ with the assertion (thick), as in cmp3. Right: shift of δ = δ − δ0 (thick), as in cmp3. '
       'Thin lines on both panels: mean δ0 = logit(context answer) − logit(prior answer) for the same question WITHOUT the assertion, '
       'averaged over the same rows; δ0 < 0 means the layer already leans to the memorised (prior) answer.',
       'J-lens only (no matrix for the last layer); x = absolute layer number. Only facts the model knows; rows grouped by the final answer WITH the assertion.']
fig.text(0.01, -0.01, '\n'.join(fill(t, 225) for t in txt), ha='left', va='top', fontsize=8.5, color=INK2, linespacing=1.4)
fig.savefig(OUT / 'cmp3_with_delta0.png', dpi=150)
print('saved', OUT / 'cmp3_with_delta0.png')
