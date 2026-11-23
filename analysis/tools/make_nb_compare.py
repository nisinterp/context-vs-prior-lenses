"""Builds and runs the comparison notebook qwen3-1.7b vs qwen3-4b.
usage: make_nb_compare.py NB FINDINGS.md SUBSET_A SUBSET_B OUTDIR"""
import sys, nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

NB, FIND, SA, SB, OUTDIR = sys.argv[1:6]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# qwen3-1.7b vs qwen3-4b: where does the model change its mind?

Same experiment, same prompts, same lenses method, both models in float32 computation:
relations author, capital, color, composer; 605 facts × 19 linguistic variants (11,495 prompts per model).
Only facts that the model knows (without the assertion it picks its memorised answer) and rows that end with the context or the prior answer.

The two models have different depth (1.7B: 28 layers, 4B: 36 layers), so layers are compared on a **relative depth** scale:
relative depth = layer / (number of layers − 1), 0 = first layer, 1 = last layer.
Colours: **violet = qwen3-1.7b**, **aqua = qwen3-4b**. Line style: **solid = context won**, **dashed = prior won**.
"""))

cells.append(code(f"""import json, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path
from textwrap import fill, wrap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
%matplotlib inline

ROOT = Path.cwd()
while not (ROOT / 'results').is_dir():
    ROOT = ROOT.parent
OUT = ROOT / {OUTDIR!r}; OUT.mkdir(parents=True, exist_ok=True)
SUBSETS = {{'qwen3-1.7b': {SA!r}, 'qwen3-4b': {SB!r}}}
"""
"""
MCOL = {'qwen3-1.7b': '#6250d6', 'qwen3-4b': '#1baf7a'}
CTX, PRI = '#2a78d6', '#e34948'
INK, INK2, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
mpl.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 2, 'font.size': 9, 'axes.titlesize': 10, 'legend.frameon': False, 'savefig.bbox': 'tight',
})
TERMS = ('Context answer = the answer stated in the assertion added to the prompt (it contradicts the fact). '
         'Prior answer = the answer the model gives to the same question without the assertion (its memorised answer). '
         '"Context won" / "prior won" = which of the two the model gives as its final output. Only facts the model knows are used.')
DELTA_T = ('δ = logit(first token of the context answer) − logit(first token of the prior answer) at one layer. '
           'Shift of δ = δ with the assertion − δ without it: how far the assertion pushes the layer toward the context answer.')
LENS_T = ('Logit lens: layer output → final norm → output matrix, as if the model stopped there. '
          'J-lens: the layer output is first multiplied by the Jacobian lens matrix of that layer (neuronpedia/jacobian-lens, fitted on wikitext), then decoded the same way. '
          'The J-lens has no matrix for the last layer.')
REL_T = 'Relative depth = layer / (number of layers − 1): qwen3-1.7b has 28 layers (0-27), qwen3-4b has 36 (0-35).'

def caption(fig, *parts, y=-0.01):
    width = int(fig.get_size_inches()[0] * 15)
    fig.text(0.01, y, '\\n'.join(fill(t, width) for t in parts), ha='left', va='top', fontsize=8.5, color=INK2, linespacing=1.4)

DIM_ORDER = ['evidentiality', 'epistemic_stance', 'form', 'tone']
D = {}
for m, path in SUBSETS.items():
    raw = pd.read_parquet(ROOT / path)
    df = raw[raw.knows_prior & raw.choice.isin(['context', 'prior'])].copy().reset_index(drop=True)
    for c in ['delta_logit_lens', 'delta_jlens', 'delta0_logit_lens', 'delta0_jlens', 'jlens_layers']:
        df[c] = df[c].map(json.loads)
    LL, JL = np.stack(df.delta_logit_lens), np.stack(df.delta_jlens)
    LL0, JL0 = np.stack(df.delta0_logit_lens), np.stack(df.delta0_jlens)
    nl = LL.shape[1]
    x_ll = np.arange(nl); x_jl = np.array(df.jlens_layers.iloc[0])
    is_ctx = (df.choice == 'context').values
    gap = LL[is_ctx].mean(0) - LL[~is_ctx].mean(0)
    sep = int(np.where(gap > 2)[0][0])
    D[m] = dict(raw=raw, df=df, LL=LL, JL=JL, S_LL=LL - LL0, S_JL=JL - JL0, nl=nl, x_ll=x_ll, x_jl=x_jl,
                r_ll=x_ll / (nl - 1), r_jl=x_jl / (nl - 1), is_ctx=is_ctx, sep=sep)
    known = raw.drop_duplicates('fact_index').knows_prior.mean()
    print(f'{m}: layers {nl} | facts {raw.fact_index.nunique()} | known facts {known:.1%} | rows used {len(df):,} '
          f'(context won {is_ctx.sum():,} = {is_ctx.mean():.1%}) | curves split (>2 logits) at layer {sep} = depth {sep/(nl-1):.2f}')
CATS = (D['qwen3-1.7b']['df'][['dimension', 'category']].drop_duplicates()
        .assign(d=lambda t: t.dimension.map(DIM_ORDER.index)).sort_values(['d', 'category'])[['dimension', 'category']])
CATS = list(CATS.itertuples(index=False, name=None))
"""))

cells.append(md("""## C1. Behaviour: how often the context wins, per linguistic type, in each model"""))
cells.append(code("""beh = []
for m in D:
    df = D[m]['df']
    g = df.groupby(['dimension', 'category']).choice.agg(n='size', ctx=lambda s: (s == 'context').mean() * 100)
    beh.append(g.add_prefix(m + ' '))
beh = pd.concat(beh, axis=1).reset_index()
beh['dim_i'] = beh.dimension.map(DIM_ORDER.index)
beh = beh.sort_values(['dim_i', 'qwen3-1.7b ctx']).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 7))
y = np.arange(len(beh))
for i, r in beh.iterrows():
    ax.plot([r['qwen3-1.7b ctx'], r['qwen3-4b ctx']], [i, i], color=BASE, lw=2, zorder=1)
for m in D:
    ax.scatter(beh[m + ' ctx'], y, s=60, color=MCOL[m], zorder=2, label=m, edgecolor='#fcfcfb', linewidth=1.5)
ax.set_yticks(y, [f'{r.category}  ({r.dimension})' for r in beh.itertuples()])
ax.axvline(50, color=BASE, lw=1); ax.set_xlabel('% of rows where the context answer won (known facts)')
ax.grid(axis='y', visible=False)
ax.set_title('Context won, per linguistic type: qwen3-1.7b vs qwen3-4b', loc='left')
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=2)
rho = beh['qwen3-1.7b ctx'].rank().corr(beh['qwen3-4b ctx'].rank())  # Spearman = Pearson on ranks
caption(fig, f'Each row = one linguistic type; dots = share of rows where the model gave the context answer. Rank correlation of the 19 types between the models (Spearman) = {rho:.2f}.', TERMS)
fig.savefig(OUT / 'cmp1_behaviour.png', dpi=150); FIGC1 = fig; plt.show()
print('Spearman rho of type ranking between models:', round(rho, 3))
beh.drop(columns='dim_i').round(1)
"""))

cells.append(md("""## C2. Lens curves of both models on one relative-depth axis

Mean shift of δ caused by the assertion, pooled over all 19 types. Left: logit lens; right: J-lens."""))
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
for ax, key, xk, ttl in [(axes[0], 'S_LL', 'r_ll', 'Logit lens'), (axes[1], 'S_JL', 'r_jl', 'J-lens')]:
    for m in D:
        d = D[m]
        ax.plot(d[xk], d[key][d['is_ctx']].mean(0), color=MCOL[m], lw=2)
        ax.plot(d[xk], d[key][~d['is_ctx']].mean(0), color=MCOL[m], lw=2, ls='--')
        ax.axvline(d['sep'] / (d['nl'] - 1), color=MCOL[m], lw=1, ls=':')
    ax.axhline(0, color=BASE, lw=1)
    ax.set_title(f'{ttl}: shift of δ caused by the assertion', loc='left'); ax.set_xlabel('relative depth (0 = first layer, 1 = last layer)')
axes[0].set_ylabel('mean shift of δ (logits)')
h = [Line2D([], [], color=MCOL[m], lw=2, label=m) for m in D] + \\
    [Line2D([], [], color=INK2, lw=2, label='context won'), Line2D([], [], color=INK2, lw=2, ls='--', label='prior won'),
     Line2D([], [], color=INK2, lw=1, ls=':', label='layer where the curves split')]
axes[0].legend(handles=h, loc='upper left')
fig.tight_layout()
caption(fig, 'Lines = mean over all rows of the group (all 19 linguistic types pooled). Dotted vertical line = first layer where the mean δ of context-won and prior-won rows differ by more than 2 logits (logit lens).',
        DELTA_T, LENS_T, REL_T, TERMS)
fig.savefig(OUT / 'cmp2_shift_relative_depth.png', dpi=150); FIGC2 = fig; plt.show()
"""))

cells.append(md("""## C3. The same curves in absolute layers, and raw δ"""))
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
for ax, key, ttl, yl in [(axes[0], 'JL', 'J-lens: mean δ (context − prior)', 'mean δ (logits)'),
                         (axes[1], 'S_JL', 'J-lens: mean shift of δ caused by the assertion', 'mean shift of δ (logits)')]:
    for m in D:
        d = D[m]
        ax.plot(d['x_jl'], d[key][d['is_ctx']].mean(0), color=MCOL[m], lw=2)
        ax.plot(d['x_jl'], d[key][~d['is_ctx']].mean(0), color=MCOL[m], lw=2, ls='--')
    ax.axhline(0, color=BASE, lw=1); ax.set_title(ttl, loc='left'); ax.set_xlabel('layer'); ax.set_ylabel(yl)
h = [Line2D([], [], color=MCOL[m], lw=2, label=m) for m in D] + \\
    [Line2D([], [], color=INK2, lw=2, label='context won'), Line2D([], [], color=INK2, lw=2, ls='--', label='prior won')]
axes[0].legend(handles=h, loc='upper left')
fig.tight_layout()
caption(fig, 'Left: raw δ (which answer leads at each layer). Rows are grouped by final answer, so the two groups must end on opposite sides of 0. '
        'Right: δ with the assertion minus δ without it (the push of the assertion). x = absolute layer number.', DELTA_T, LENS_T, TERMS)
fig.savefig(OUT / 'cmp3_jlens_absolute_layers.png', dpi=150); FIGC3 = fig; plt.show()
"""))

cells.append(md("""## C4. Key layers per linguistic type, on relative depth"""))
cells.append(code("""def mean_ci(v):
    v = v.dropna(); return v.mean(), (1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)

metrics = [('Divergence layer, J-lens (all rows)', 'divergence_jl', None),
           ('Flip layer, J-lens (rows where context won)', 'flip_jlens', True)]
order = list(beh[['dimension', 'category']].itertuples(index=False, name=None))
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
y = np.arange(len(order))
KEY = []
for ax, (ttl, col, won) in zip(axes, metrics):
    for k, m in enumerate(D):
        d = D[m]; df = d['df']
        for i, (dim, cat) in enumerate(order):
            msk = (df.dimension == dim) & (df.category == cat)
            if won is not None: msk &= (df.choice == 'context') == won
            mu, ci = mean_ci(df.loc[msk, col] / (d['nl'] - 1))
            ax.errorbar(mu, i + (k - 0.5) * 0.3, xerr=ci, fmt='o', ms=5, color=MCOL[m], ecolor=MCOL[m], elinewidth=1, capsize=2,
                        label=m if i == 0 else None)
            KEY.append(dict(model=m, metric=col, dimension=dim, category=cat, rel=mu, layer=mu * (d['nl'] - 1)))
    ax.set_title(ttl, loc='left'); ax.set_xlabel('relative depth'); ax.grid(axis='y', visible=False)
axes[0].set_yticks(y, [f'{c} [{d}]' for d, c in order]); axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.06), ncol=2)
fig.tight_layout()
caption(fig, 'Dot = mean over the rows of a type, as relative depth; bar = 95% confidence interval. Types are in the same order as in C1.',
        'Divergence layer = first layer from which δ with the assertion stays more than 2 logits above δ without it. Flip layer = first layer from which δ stays above 0 to the end.',
        REL_T, LENS_T, TERMS)
fig.savefig(OUT / 'cmp4_key_layers_relative.png', dpi=150); FIGC4 = fig; plt.show()
KEY = pd.DataFrame(KEY)
"""))

cells.append(md("""## C5. Do the same types win in both models?"""))
cells.append(code("""fig, ax = plt.subplots(figsize=(7.5, 7))
ax.scatter(beh['qwen3-1.7b ctx'], beh['qwen3-4b ctx'], s=50, color=INK2, zorder=2)
for _, r in beh.iterrows():
    ax.annotate(r['category'], (r['qwen3-1.7b ctx'], r['qwen3-4b ctx']), fontsize=7.5, color=INK2, xytext=(4, 2), textcoords='offset points')
lo = min(beh['qwen3-1.7b ctx'].min(), beh['qwen3-4b ctx'].min()) - 5; hi = max(beh['qwen3-1.7b ctx'].max(), beh['qwen3-4b ctx'].max()) + 5
ax.plot([lo, hi], [lo, hi], color=BASE, lw=1, ls='--'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel('qwen3-1.7b: % context won'); ax.set_ylabel('qwen3-4b: % context won')
ax.set_title('Context won per linguistic type: 1.7B (x) vs 4B (y)', loc='left')
caption(fig, f'One dot per linguistic type. Dashed line = equal in both models; below it = the 4B model follows the context less often. Spearman rank correlation = {rho:.2f}.', TERMS)
fig.savefig(OUT / 'cmp5_types_scatter.png', dpi=150); FIGC5 = fig; plt.show()
"""))

cells.append(md("""## C6. Summary table"""))
cells.append(code("""rows = []
for dim, cat in order:
    r = {'dimension': dim, 'category': cat}
    for m in D:
        d = D[m]; df = d['df']; msk = ((df.dimension == dim) & (df.category == cat)).values
        r[f'{m} n'] = int(msk.sum())
        r[f'{m} ctx %'] = 100 * d['is_ctx'][msk].mean()
        r[f'{m} div JL (layer)'] = df.loc[msk, 'divergence_jl'].mean()
        r[f'{m} div JL (rel)'] = df.loc[msk, 'divergence_jl'].mean() / (d['nl'] - 1)
        pk = int(np.argmax(d['S_JL'][d['is_ctx']].mean(0)))
        r[f'{m} max shift JL (ctx)'] = d['S_JL'][msk & d['is_ctx'], pk].mean()
    rows.append(r)
TAB = pd.DataFrame(rows).round(2)
TAB.to_csv(OUT / 'comparison_by_category.csv', index=False)
for m in D:
    d = D[m]; pk = int(np.argmax(d['S_JL'][d['is_ctx']].mean(0)))
    print(f"{m}: J-lens shift peaks at layer {d['x_jl'][pk]} (depth {d['x_jl'][pk]/(d['nl']-1):.2f}); "
          f"context-won {d['S_JL'][d['is_ctx'], pk].mean():.1f}, prior-won {d['S_JL'][~d['is_ctx'], pk].mean():.1f} logits; "
          f"split at layer {d['sep']} (depth {d['sep']/(d['nl']-1):.2f}); "
          f"median divergence JL {d['df'].divergence_jl.median()} (depth {d['df'].divergence_jl.median()/(d['nl']-1):.2f}); "
          f"median flip JL (ctx) {d['df'].loc[d['is_ctx'], 'flip_jlens'].median()}")
    early = d['sep'] - 1
    print(f"   mean |shift| before layer {early}: LL {np.abs(d['S_LL'][:, :early].mean(0)).max():.2f}, JL {np.abs(d['S_JL'][:, :early].mean(0)).max():.2f}")
    tt = TAB.set_index('category')[f'{m} ctx %']
    print('   corr(ctx %, divergence layer) across types:', round(tt.corr(TAB.set_index('category')[f'{m} div JL (layer)']), 2),
          '| corr(ctx %, max shift):', round(tt.corr(TAB.set_index('category')[f'{m} max shift JL (ctx)']), 2))
TAB
"""))

cells.append(md("## Findings\n\nFINDINGS_TEXT"))
cells.append(code("""def text_page(pdf, title, paragraphs):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.07, 0.95, title, fontsize=15, weight='bold', va='top')
    yy = 0.91
    for p in paragraphs:
        for line in wrap(p, 95) or ['']:
            if yy < 0.05:  # page full: continue on a new page
                pdf.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69)); yy = 0.95
            fig.text(0.07, yy, line, fontsize=9.5, va='top'); yy -= 0.018
        yy -= 0.012
    pdf.savefig(fig); plt.close(fig)

PDF = OUT / 'qwen3-1.7b_vs_4b_report.pdf'
with PdfPages(PDF) as pdf:
    text_page(pdf, 'qwen3-1.7b vs qwen3-4b: where the model changes its mind', REPORT_TEXT)
    for f in [FIGC1, FIGC2, FIGC3, FIGC4, FIGC5]:
        pdf.savefig(f, bbox_inches='tight')
    show = TAB[['dimension', 'category'] + [c for c in TAB.columns if ('ctx %' in c or 'rel' in c or 'max shift' in c)]]
    fig, ax = plt.subplots(figsize=(16, 7)); ax.axis('off')
    t = ax.table(cellText=show.values, colLabels=['\\n'.join(wrap(c, 14)) for c in show.columns], loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        if r == 0: cell.set_height(cell.get_height() * 2.2)
    ax.set_title('Per linguistic type: % context won, divergence layer (relative depth), largest J-lens shift (logits, context-won rows)', loc='left')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
print('saved', PDF.relative_to(ROOT))
"""))

TXT = open(FIND).read() if Path(FIND).exists() else 'To be written.'
cells[-2] = md('## Findings\n\n' + TXT)
paras = [p.replace('**', '').replace('$', '').lstrip('- ') for p in TXT.split('\n') if p.strip() and not p.startswith('#')]
cells[-1].source = 'REPORT_TEXT = ' + repr(paras) + '\n' + cells[-1].source
nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(Path(NB).resolve().parent)}}).execute()
nbf.write(nb, NB)
print('ok', NB)
