"""
Shift caused by the assertion (δ − δ0) vs strength of the fact (−δ0), one point per row, one panel per
linguistic type; the context wins exactly above the diagonal shift = strength. Last layer, logit lens,
known facts only. The coloured line is the median shift in 8 equal-size strength bins; the last panel
overlays these medians for all 19 types. One figure per model. Uses only the saved analysis subsets
(no model).

Usage (from the repo root):
    python analysis/plot_shift_vs_strength_scatter.py
Writes results/comparison/shift_vs_strength_qwen3-1.7b.png and shift_vs_strength_qwen3-4b.png
(or --out DIR).
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
CTX, PRI = '#2a78d6', '#d6452a'
INK2, MUTED, GRID, BASE = '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
mpl.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'font.size': 8.5, 'axes.titlesize': 9, 'legend.frameon': False, 'savefig.bbox': 'tight'})
cmap = mpl.colormaps['viridis_r']; norm = mpl.colors.Normalize(45, 85)
DATA = {}
for m, p in SUBSETS.items():
    raw = pd.read_parquet(ROOT / p)
    df = raw[raw.knows_prior & raw.choice.isin(['context', 'prior'])].reset_index(drop=True)
    last = lambda c: df[c].map(lambda v: json.loads(v)[-1]).values
    df = df.assign(strength=-last('delta0_logit_lens'), shift=last('delta_logit_lens') - last('delta0_logit_lens'),
                   ctx=(df.choice == 'context').values)
    DATA[m] = df
lim = max(max(d.strength.quantile(0.995), d['shift'].quantile(0.995)) for d in DATA.values())
lo = min(d['shift'].quantile(0.002) for d in DATA.values())
for m, df in DATA.items():
    pct = df.groupby('category').ctx.mean().mul(100).sort_values(ascending=False)
    bins = np.quantile(df.strength, np.linspace(0, 1, 9))
    fig, axes = plt.subplots(5, 4, figsize=(17, 20), sharex=True, sharey=True)
    summ = axes.flat[19]
    for ax, (cat, v) in zip(axes.flat, pct.items()):
        t = df[df.category == cat]
        ax.scatter(t.strength[~t.ctx], t['shift'][~t.ctx], s=6, color=PRI, alpha=0.45, lw=0)
        ax.scatter(t.strength[t.ctx], t['shift'][t.ctx], s=6, color=CTX, alpha=0.45, lw=0)
        ax.plot([0, lim], [0, lim], color=INK2, lw=1)
        # binned median shift (8 equal-size bins of fact strength over all rows of this model)
        mid = [(t.strength[(t.strength >= a) & (t.strength < b)].median(), t['shift'][(t.strength >= a) & (t.strength < b)].median())
               for a, b in zip(bins[:-1], bins[1:])]
        mid = np.array(mid)
        ax.plot(mid[:, 0], mid[:, 1], color=cmap(norm(v)), lw=2.2, marker='o', ms=3.5)
        summ.plot(mid[:, 0], mid[:, 1], color=cmap(norm(v)), lw=1.4)
        ax.set_title(f'{cat}: context won {v:.0f}% ({t.ctx.sum()}/{len(t)}), median shift {t["shift"].median():.1f}')
    summ.plot([0, lim], [0, lim], color=INK2, lw=1)
    summ.set_title('all 19 types: median shift per strength bin\n(colour = % context won)')
    for ax in axes[-1]: ax.set_xlabel('strength of the fact = −δ0 (logits)')
    for ax in axes[:, 0]: ax.set_ylabel('shift caused by the assertion (logits)')
    axes.flat[0].set_xlim(0, lim); axes.flat[0].set_ylim(lo, lim)
    h = [Line2D([], [], color=CTX, marker='o', ls='', ms=5, label='row where the context won'),
         Line2D([], [], color=PRI, marker='o', ls='', ms=5, label='row where the prior won'),
         Line2D([], [], color=INK2, lw=1, label='shift = strength (the context wins above this line)'),
         Line2D([], [], color=cmap(norm(65)), lw=2.2, marker='o', ms=3.5, label='median shift in 8 equal-size strength bins')]
    fig.legend(handles=h, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.01), fontsize=10)
    fig.suptitle(f'{m}: shift caused by the assertion vs strength of the fact, per linguistic type '
                 f'({df.fact_index.nunique()} known facts, last layer, logit lens)', x=0.01, ha='left', fontsize=12, y=1.025)
    fig.tight_layout()
    txt = ['Each point = one row (fact × linguistic type). x = −δ0: how strongly the model prefers the memorised answer WITHOUT the assertion '
           '(same for all 19 types of a fact). y = δ − δ0: how far the assertion moves δ. The context wins exactly when δ = δ0 + shift > 0, '
           'i.e. above the diagonal. Types sorted by % context won. Axes cut at the 99.5% quantile; a few extreme points lie outside.']
    fig.text(0.01, -0.003, '\n'.join(fill(t, 230) for t in txt), ha='left', va='top', fontsize=8.5, color=INK2, linespacing=1.4)
    fig.savefig(OUT / f'shift_vs_strength_{m}.png', dpi=110)
    print('saved', m, 'rows', len(df), 'ctx-won above diagonal:', ((df['shift'] > df.strength) == df.ctx).mean())
