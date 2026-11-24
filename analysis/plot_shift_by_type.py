"""
Mean shift of δ caused by the assertion (δ − δ0) over ALL rows of each linguistic type, not split by
the outcome. One line per type, coloured by the % of rows of that type where the context won; panels:
logit lens and J-lens, Qwen3-1.7B and Qwen3-4B. Uses only the saved analysis subsets (no model).
Also prints, per panel, the correlation between % context won and the shift at the last layer / peak.

Usage (from the repo root):
    python analysis/plot_shift_by_type.py
Writes results/comparison/shift_all_rows_by_type.png (or --out DIR).
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
args = ap.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
SUBSETS = {'qwen3-1.7b': 'results/qwen3-1.7b/qwen3-1.7b_4relations.parquet',
           'qwen3-4b': 'results/qwen3-4b/qwen3-4b_4relations.parquet'}
INK, INK2, MUTED, GRID, BASE = '#2b2a27', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
mpl.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'font.size': 9, 'axes.titlesize': 10, 'legend.frameon': False, 'savefig.bbox': 'tight'})
cmap = mpl.colormaps['viridis_r']; norm = mpl.colors.Normalize(45, 85)
R = []
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
for j, (m, p) in enumerate(SUBSETS.items()):
    raw = pd.read_parquet(ROOT / p)
    df = raw[raw.knows_prior & raw.choice.isin(['context', 'prior'])].reset_index(drop=True)
    st = lambda c: np.stack(df[c].map(json.loads))
    S = {'logit lens': (np.arange(st('delta_logit_lens').shape[1]), st('delta_logit_lens') - st('delta0_logit_lens')),
         'J-lens': (np.array(json.loads(df.jlens_layers.iloc[0])), st('delta_jlens') - st('delta0_jlens'))}
    pct = df.groupby('category').choice.apply(lambda s: 100 * (s == 'context').mean()).sort_values()
    for i, (lens, (x, sh)) in enumerate(S.items()):
        ax = axes[i, j]
        ends = []
        for cat, v in pct.items():
            y = sh[(df.category == cat).values].mean(0)
            ax.plot(x, y, color=cmap(norm(v)), lw=1.6)
            ends.append((y[-1], f'{cat} {v:.0f}%', v)); R.append((m, lens, cat, v, y[-1], y.max()))
        ends.sort()
        # direct labels: the 3 lowest and 3 highest end points, spread vertically
        lab = ends[:3] + ends[-3:]
        lo, hi = ax.get_ylim(); gap = (hi - lo) * 0.035
        ys = [e[0] for e in lab]
        for k in range(1, len(ys)): ys[k] = max(ys[k], ys[k - 1] + gap)
        for (y0, t, v), yl in zip(lab, ys):
            ax.annotate(t, (x[-1], y0), xytext=(x[-1] + 0.6, yl), fontsize=7.5, color=INK2, va='center',
                        arrowprops=dict(arrowstyle='-', color=BASE, lw=0.6))
        ax.set_xlim(x[0], x[-1] + 6)
        ax.axhline(0, color=BASE, lw=1)
        ax.set_title(f'{m}, {lens}: mean shift δ − δ0 over ALL rows of each type', loc='left')
        ax.set_xlabel('layer'); ax.set_ylabel('mean shift (logits)')
fig.subplots_adjust(left=0.05, right=0.88, hspace=0.3, wspace=0.22)
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0.91, 0.3, 0.012, 0.4]))
cb.set_label('% rows where the context won (per type, per model)', color=INK2)
txt = ['Each line = one of the 19 linguistic types; mean of δ(with the assertion) − δ0(without) over all rows of that type (known facts, '
       'final answer = context or prior), NOT split by the outcome. Line colour = % of rows of that type where the context won. '
       'Labels: the 3 types with the lowest and the 3 with the highest shift at the last layer shown. J-lens has no matrix for the last layer.']
fig.text(0.05, 0.05, '\n'.join(fill(t, 200) for t in txt), ha='left', va='top', fontsize=8.5, color=INK2, linespacing=1.4)
fig.savefig(OUT / 'shift_all_rows_by_type.png', dpi=130)
print('saved', OUT / 'shift_all_rows_by_type.png')
r = pd.DataFrame(R, columns=['m','lens','cat','pct','end','peak'])
for (m, l), g in r.groupby(['m','lens'], sort=False):
    print(m, l, 'corr(%%context won, shift at last layer)=%.2f, corr(.., peak shift)=%.2f; last-layer shift %.1f..%.1f, peak %.1f..%.1f' % (g.pct.corr(g.end), g.pct.corr(g.peak), g.end.min(), g.end.max(), g.peak.min(), g.peak.max()))
