"""
Mean assertion shift (δ − δ0) over all rows of each linguistic type, split by the strength of the fact.
Only facts known by both models; strong = δ0 < --th (question without the assertion, last layer,
logit lens) in both models, weak = the other facts known by both. Rows: strong / weak facts; columns:
Qwen3-1.7B and Qwen3-4B, logit lens and J-lens. Uses only the saved analysis subsets (no model).

Usage (from the repo root):
    python analysis/plot_shift_strong_weak.py
Writes results/comparison/shift_by_type_strong_weak.png (or --out DIR).
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
ap.add_argument('--th', type=float, default=-30.0, help='delta0 threshold for a strong fact (default -30)')
args = ap.parse_args()
OUT, TH = args.out, args.th
OUT.mkdir(parents=True, exist_ok=True)
SUBSETS = {'qwen3-1.7b': 'results/qwen3-1.7b/qwen3-1.7b_4relations.parquet',
           'qwen3-4b': 'results/qwen3-4b/qwen3-4b_4relations.parquet'}
INK2, MUTED, GRID, BASE = '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
mpl.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'font.size': 8.5, 'axes.titlesize': 9, 'legend.frameon': False, 'savefig.bbox': 'tight'})
cmap = mpl.colormaps['viridis_r']
last = lambda v: json.loads(v)[-1]
RAW = {m: pd.read_parquet(ROOT / p) for m, p in SUBSETS.items()}
d0 = pd.concat({m: r[r.knows_prior].drop_duplicates('fact_index').set_index('fact_index').delta0_logit_lens.map(last)
                for m, r in RAW.items()}, axis=1, join='inner')
STRONG = set(d0.index[(d0 < TH).all(axis=1)]); WEAK = set(d0.index) - STRONG
GROUPS = {'strong': STRONG, 'weak': WEAK}
print('known by both', len(d0), 'strong', len(STRONG), 'weak', len(WEAK))
# panels: rows = strong / weak, columns = (model, lens)
C = [(m, l) for m in SUBSETS for l in ('logit lens', 'J-lens')]
fig, axes = plt.subplots(2, 4, figsize=(21, 11), sharey=True)
for i, (gname, facts) in enumerate(GROUPS.items()):
    pcts = {}
    for m, raw in RAW.items():
        df = raw[raw.fact_index.isin(facts) & raw.choice.isin(['context', 'prior'])]
        pcts[m] = df.groupby('category').choice.apply(lambda s: 100 * (s == 'context').mean())
    allp = pd.concat(pcts.values()); norm = mpl.colors.Normalize(np.floor(allp.min() / 5) * 5, np.ceil(allp.max() / 5) * 5)
    for j, (m, lens) in enumerate(C):
        ax = axes[i, j]; raw = RAW[m]
        df = raw[raw.fact_index.isin(facts) & raw.choice.isin(['context', 'prior'])].reset_index(drop=True)
        st = lambda c: np.stack(df[c].map(json.loads))
        if lens == 'logit lens':
            sh = st('delta_logit_lens') - st('delta0_logit_lens'); x = np.arange(sh.shape[1])
        else:
            sh = st('delta_jlens') - st('delta0_jlens'); x = np.array(json.loads(df.jlens_layers.iloc[0]))
        pct = pcts[m].sort_values(); ends = []
        for cat, v in pct.items():
            y = sh[(df.category == cat).values].mean(0)
            ax.plot(x, y, color=cmap(norm(v)), lw=1.5)
            ends.append((y[-1], f'{cat} {v:.0f}%'))
        ends.sort(); ends = ends[:3] + ends[-3:]
        ys = [e[0] for e in ends]
        for k in range(1, len(ys)): ys[k] = max(ys[k], ys[k - 1] + 2.4)
        for (y0, t), yl in zip(ends, ys):
            ax.annotate(t, (x[-1], y0), xytext=(x[-1] + 0.6, yl), fontsize=7, color=INK2, va='center',
                        arrowprops=dict(arrowstyle='-', color=BASE, lw=0.6))
        ax.set_xlim(x[0], x[-1] + 8); ax.axhline(0, color=BASE, lw=1)
        nf = df.fact_index.nunique()
        ax.set_title(f'{m}, {lens} — {gname} facts ({nf})\ncontext won {100 * (df.choice == "context").mean():.0f}% of rows; '
                     f'types {pct.min():.0f}–{pct.max():.0f}%', loc='left')
        if i == 1: ax.set_xlabel('layer')
        if j == 0: ax.set_ylabel(f'{gname} facts: mean shift δ − δ0 (logits)')
    fig.subplots_adjust(left=0.04, right=0.9, hspace=0.32, wspace=0.12)
    pos = axes[i, -1].get_position()
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0.925, pos.y0 + 0.03, 0.008, pos.height - 0.06]))
    cb.set_label(f'% context won per type\n({gname} facts)', color=INK2)
txt = [f'Same as shift_all_rows_by_type.png, split by the strength of the fact. Only facts known by BOTH models. Strong = δ0 < {TH:g} '
       '(question without the assertion, last layer, logit lens) in both models; weak = all the other facts known by both. '
       'Each line = one linguistic type, mean of δ − δ0 over all its rows (not split by the outcome); colour = % context won of that type '
       'within this group of facts (separate colour scale per row). Labels: 3 lowest and 3 highest at the last layer. Same y-axis on all panels.']
fig.text(0.04, 0.03, '\n'.join(fill(t, 230) for t in txt), ha='left', va='top', fontsize=8.5, color=INK2, linespacing=1.4)
fig.savefig(OUT / 'shift_by_type_strong_weak.png', dpi=120)
print('saved', OUT / 'shift_by_type_strong_weak.png')
