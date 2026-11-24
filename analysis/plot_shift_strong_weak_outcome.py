"""
Mean assertion shift (δ − δ0) per linguistic type, split by the strength of the fact AND by the final
answer with the assertion (context won / prior won). Only facts known by both models; strong = δ0 < --th
(question without the assertion, last layer, logit lens) in both models, weak = the others. Rows:
strong/context won, strong/prior won, weak/context won, weak/prior won; columns: Qwen3-1.7B and
Qwen3-4B, logit lens and J-lens. Types with fewer than --min-rows rows in a panel are not drawn.
Uses only the saved analysis subsets (no model); prints the last-layer range per panel (logit lens).

Usage (from the repo root):
    python analysis/plot_shift_strong_weak_outcome.py
Writes results/comparison/shift_by_type_strong_weak_outcome.png (or --out DIR).
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
ap.add_argument('--min-rows', type=int, default=5, help='do not draw a type with fewer rows in a panel (default 5)')
args = ap.parse_args()
OUT, TH, MIN_ROWS = args.out, args.th, args.min_rows
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
C = [(m, l) for m in SUBSETS for l in ('logit lens', 'J-lens')]
ROWS = [(g, o) for g in GROUPS for o in ('context', 'prior')]
fig, axes = plt.subplots(4, 4, figsize=(21, 21), sharey=True)
fig.subplots_adjust(left=0.05, right=0.9, hspace=0.36, wspace=0.12)
for gname, facts in GROUPS.items():
    pcts = {m: raw[raw.fact_index.isin(facts) & raw.choice.isin(['context', 'prior'])]
            .groupby('category').choice.apply(lambda s: 100 * (s == 'context').mean()) for m, raw in RAW.items()}
    allp = pd.concat(pcts.values()); norm = mpl.colors.Normalize(np.floor(allp.min() / 5) * 5, np.ceil(allp.max() / 5) * 5)
    for outcome in ('context', 'prior'):
        i = ROWS.index((gname, outcome))
        for j, (m, lens) in enumerate(C):
            ax = axes[i, j]; raw = RAW[m]
            df = raw[raw.fact_index.isin(facts) & (raw.choice == outcome)].reset_index(drop=True)
            st = lambda c: np.stack(df[c].map(json.loads))
            if lens == 'logit lens':
                sh = st('delta_logit_lens') - st('delta0_logit_lens'); x = np.arange(sh.shape[1])
            else:
                sh = st('delta_jlens') - st('delta0_jlens'); x = np.array(json.loads(df.jlens_layers.iloc[0]))
            ends, skipped = [], []
            for cat, v in pcts[m].sort_values().items():
                sel = (df.category == cat).values
                if sel.sum() < MIN_ROWS: skipped.append(cat); continue
                y = sh[sel].mean(0)
                ax.plot(x, y, color=cmap(norm(v)), lw=1.5)
                ends.append((y[-1], f'{cat} {v:.0f}% (n={sel.sum()})'))
            ends.sort(); ends = ends[:3] + ends[-3:] if len(ends) > 6 else ends
            ys = [e[0] for e in ends]
            for k in range(1, len(ys)): ys[k] = max(ys[k], ys[k - 1] + 3.0)
            for (y0, t), yl in zip(ends, ys):
                ax.annotate(t, (x[-1], y0), xytext=(x[-1] + 0.6, yl), fontsize=6.8, color=INK2, va='center',
                            arrowprops=dict(arrowstyle='-', color=BASE, lw=0.6))
            ax.set_xlim(x[0], x[-1] + 11); ax.axhline(0, color=BASE, lw=1)
            sk = f'\n({len(skipped)} types with <{MIN_ROWS} rows not drawn)' if skipped else ''
            ax.set_title(f'{m}, {lens}\n{gname} facts, rows where the {outcome} won: {len(df)}{sk}', loc='left')
            if lens == 'logit lens': print(gname, outcome, m, 'last-layer shift per type %.1f..%.1f' % (min(e[0] for e in ends), max(e[0] for e in ends)), 'pooled %.1f' % sh[:, -1].mean())
            if i == 3: ax.set_xlabel('layer')
            if j == 0: ax.set_ylabel(f'{gname} facts, {outcome} won:\nmean shift δ − δ0 (logits)')
    p0, p1 = axes[ROWS.index((gname, 'prior')), -1].get_position(), axes[ROWS.index((gname, 'context')), -1].get_position()
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0.935, p0.y0 + 0.04, 0.008, p1.y1 - p0.y0 - 0.08]))
    cb.set_label(f'% context won per type, over all its rows\n({gname} facts)', color=INK2)
txt = [f'Same as shift_by_type_strong_weak.png, each panel split by the final answer WITH the assertion (context won / prior won). '
       f'Facts known by both models; strong = δ0 < {TH:g} (last layer, logit lens) in both models, weak = the others. '
       'Each line = one linguistic type, mean of δ − δ0 over its rows in that group; colour = % context won of that type over ALL its rows '
       f'of this strength (so the colour is the same in the context-won and prior-won rows). Types with fewer than {MIN_ROWS} rows in a panel are not drawn. '
       'Labels: 3 lowest and 3 highest at the last layer, with the number of rows. Same y-axis on all panels.']
fig.text(0.05, 0.075, '\n'.join(fill(t, 235) for t in txt), ha='left', va='top', fontsize=8.5, color=INK2, linespacing=1.4)
fig.savefig(OUT / 'shift_by_type_strong_weak_outcome.png', dpi=110)
print('saved', OUT / 'shift_by_type_strong_weak_outcome.png')
