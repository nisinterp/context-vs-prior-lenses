"""Builds and runs a single-model analysis notebook.
usage (from analysis/): make_nb_model.py NB FINDINGS.md MODEL SUBSET RESDIR   (SUBSET, RESDIR relative to the repo root)"""
import sys, nbformat as nbf
from nbclient import NotebookClient

NB = sys.argv[1]
MODEL, SUBSET, RESDIR = sys.argv[3], sys.argv[4], sys.argv[5]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(f"""# Where does {MODEL} change its mind? Logit lens and J-lens by linguistic parameter

**Data:** `{SUBSET}` (relations author, capital, color, composer; 605 facts × 19 linguistic variants).

**Rows used:** only facts the model already knows (`knows_prior=True`, i.e. without the assertion it picks its own memorised answer), and only rows where the final answer is the context answer or the prior answer (distractor wins are dropped). Here context and memory really compete.

**Measure:** for every layer, $\\delta_l$ = logit(first token of the context answer) − logit(first token of the prior answer).
$\\delta>0$ means the layer leans to the context answer, $\\delta<0$ to the memorised one.
Two lenses read each layer: the **logit lens** (layer output → final norm → unembedding; every layer) and the **J-lens** (layer output multiplied by the Jacobian lens first; every layer except the last).

**Layer metrics from `run_flip.py`:**
- *flip layer*: first layer after which $\\delta>0$ stays positive to the end;
- *divergence layer*: first layer after which the curve with the assertion is above the curve without it (the baseline prompt) by more than 2 logits, to the end;
- *peak prior layer*: layer where $\\delta$ is lowest, i.e. where the memorised answer is strongest.
"""))

cells.append(code("""import json, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
%matplotlib inline

MODEL, SUBSET, RESDIR = 'MODEL_X', 'SUBSET_X', 'RESDIR_X'
ROOT = Path.cwd()
while not (ROOT / 'results').is_dir():
    ROOT = ROOT.parent
RES = ROOT / RESDIR; RES.mkdir(parents=True, exist_ok=True)
OUT = RES / 'figures'; OUT.mkdir(parents=True, exist_ok=True)

# palette: blue = context won, red = prior won (diverging pair); ink and grid tokens
CTX, PRI = '#2a78d6', '#e34948'
INK, INK2, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
mpl.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 2, 'font.size': 9, 'axes.titlesize': 10, 'legend.frameon': False,
    'savefig.bbox': 'tight',
})

# --- caption texts shared by the figures ---
from textwrap import fill
TERMS = ('Context answer = the answer stated in the assertion that is added to the prompt; it contradicts what the model knows. '
         'Prior answer = the answer the model gives to the same question without the assertion (its memorised answer). '
         'Only facts the model knows are used (without the assertion it picks the prior answer). '
         '"Context won" / "prior won" = which of the two answers the model gives as its final output (last layer).')
DELTA_T = ('δ = logit of the first token of the context answer minus logit of the first token of the prior answer, read at one layer. '
           'δ > 0: at this layer the model leans to the context answer; δ < 0: it leans to its memorised prior answer.')
LENS_T = ('Logit lens (solid lines): the hidden state after layer l goes straight through the final norm and the output matrix, '
          'as if the model stopped at layer l (layers 0-{nl}). '
          'J-lens (dashed lines): the hidden state is first multiplied by the Jacobian lens matrix J_l '
          '({model} Jacobian lens from neuronpedia/jacobian-lens, fitted on wikitext), which maps layer l to what the last layer would contain, and is then decoded the same way (layers 0-{nj}).')

def caption(fig, *parts, y=-0.01):
    width = int(fig.get_size_inches()[0] * 15)
    txt = '\\n'.join(fill(t, width) for t in parts)
    fig.text(0.01, y, txt, ha='left', va='top', fontsize=8.5, color=INK2, linespacing=1.4)

raw = pd.read_parquet(ROOT / SUBSET)
df = raw[raw.knows_prior & raw.choice.isin(['context', 'prior'])].copy()
for c in ['delta_logit_lens', 'delta_jlens', 'delta0_logit_lens', 'delta0_jlens', 'jlens_layers']:
    df[c] = df[c].map(json.loads)
LL, JL = np.stack(df.delta_logit_lens), np.stack(df.delta_jlens)
LL0, JL0 = np.stack(df.delta0_logit_lens), np.stack(df.delta0_jlens)
X_LL, X_JL = np.arange(LL.shape[1]), np.array(df.jlens_layers.iloc[0])
is_ctx = (df.choice == 'context').values
df = df.reset_index(drop=True)
NL = len(X_LL)
LENS_T = LENS_T.format(nl=X_LL[-1], nj=X_JL[-1], model=MODEL)
n_dtr = int((raw.knows_prior & (raw.choice == 'distractor')).sum())
DATA_T = (f'Model: {MODEL}. Data: relations author, capital, color, composer; {raw.fact_index.nunique()} facts x 19 linguistic variants; '
          f'{len(df):,} rows with known facts ({is_ctx.sum():,} context won, {(~is_ctx).sum():,} prior won).')
# turning window: first layer where context-won and prior-won mean curves (logit lens) differ by > 2 logits, +-1 layer
_gap = LL[is_ctx].mean(0) - LL[~is_ctx].mean(0)
SEP = int(np.where(_gap > 2)[0][0])
BAND = (SEP - 1, SEP + 1)

DIM_ORDER = ['evidentiality', 'epistemic_stance', 'form', 'tone']
CATS = (df[['dimension', 'category']].drop_duplicates()
        .assign(d=lambda t: t.dimension.map(DIM_ORDER.index)).sort_values(['d', 'category'])
        [['dimension', 'category']].itertuples(index=False, name=None))
CATS = list(CATS)
print(f'all rows: {len(raw)} | known facts: {raw.knows_prior.sum()} | used (context/prior won): {len(df)}')
print(f'context won: {is_ctx.sum()} | prior won: {(~is_ctx).sum()} | layers: logit lens {len(X_LL)}, J-lens {len(X_JL)}')
"""))

cells.append(md("""## 1. Behaviour: how often the assertion wins, by linguistic parameter"""))
cells.append(code("""beh = (df.groupby(['dimension', 'category'])
       .agg(n=('id', 'size'), context_won=('choice', lambda s: (s == 'context').mean()))
       .reset_index())
beh['dim_i'] = beh.dimension.map(DIM_ORDER.index)
beh = beh.sort_values(['dim_i', 'context_won']).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 6.2))
y = np.arange(len(beh))
ax.barh(y, beh.context_won * 100, color=CTX, height=0.62, label='context won: final answer = context answer (from the assertion)')
ax.barh(y, (1 - beh.context_won) * 100, left=beh.context_won * 100, color=PRI, height=0.62, label='prior won: final answer = prior answer (memorised)')
for i, r in beh.iterrows():
    ax.text(r.context_won * 100 - 1.5, i, f'{r.context_won*100:.0f}%', va='center', ha='right', color='white', fontsize=8)
ax.set_yticks(y, [f'{r.category}  ({r.dimension})' for r in beh.itertuples()])
ax.axvline(50, color=BASE, lw=1)
ax.set_xlim(0, 100); ax.set_xlabel('% of rows (known facts)'); ax.grid(axis='y', visible=False)
ax.set_title(f'Final answer by linguistic parameter ({MODEL}, known facts)', loc='left')
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=1)
caption(fig, f'Each bar = 100% of the rows of one linguistic category ({beh.n.min()}-{beh.n.max()} rows). Distractor answers ({n_dtr} rows, {100*n_dtr/raw.knows_prior.sum():.1f}%) are left out.', TERMS, DATA_T)
fig.tight_layout(); fig.savefig(OUT / 'fig1_behaviour.png', dpi=160); FIG1 = fig; plt.show()
beh[['dimension', 'category', 'n', 'context_won']].round(3)
"""))

cells.append(md("""## 2. Lens curves by linguistic parameter (same layout as the example)

Each panel is one linguistic category. **Blue**: rows where the model finally chose the context answer; **red**: rows where it kept its memorised answer. **Solid**: logit lens; **dashed**: J-lens. y = mean $\\delta$ (context − prior logit) at each layer."""))
cells.append(code("""def lens_grid(Y_ll, Y_jl, title, fname, ylabel, cap):
    fig, axes = plt.subplots(5, 4, figsize=(15, 17), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (dim, cat) in zip(axes, CATS):
        m = (df.dimension == dim).values & (df.category == cat).values
        for won, col in [(True, CTX), (False, PRI)]:
            s = m & (is_ctx == won)
            ax.plot(X_LL, Y_ll[s].mean(0), color=col, lw=2)
            ax.plot(X_JL, Y_jl[s].mean(0), color=col, lw=2, ls='--')
        ax.axhline(0, color=BASE, lw=1)
        ax.axvspan(BAND[0] - 0.5, BAND[1] + 0.5, color='#f0efec', zorder=0)
        ax.set_title(f'{cat} [{dim}]\\ncontext won {(m & is_ctx).sum()}, prior won {(m & ~is_ctx).sum()}', fontsize=9)
    for ax in axes[len(CATS):]:
        ax.axis('off')
    handles = [Line2D([], [], color=CTX, lw=2, label='context won\\n(final answer = context answer\\nfrom the assertion)'),
               Line2D([], [], color=PRI, lw=2, label='prior won\\n(final answer = prior answer,\\nthe memorised one)'),
               Line2D([], [], color=INK2, lw=2, label=f'logit lens (layers 0-{X_LL[-1]})'),
               Line2D([], [], color=INK2, lw=2, ls='--', label=f'J-lens, Jacobian lens\\n(layers 0-{X_JL[-1]})'),
               mpl.patches.Patch(color='#f0efec', label=f'layers {BAND[0]}-{BAND[1]}:\\nwhere the curves split')]
    axes[len(CATS)].legend(handles=handles, loc='center', fontsize=10, labelspacing=1.1)
    for ax in axes[16:]: ax.set_xlabel('layer')
    for ax in axes[::4]: ax.set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha='left', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    caption(fig, *cap)
    fig.savefig(OUT / fname, dpi=130)
    return fig

FIG2 = lens_grid(LL, JL, f'Mean δ (context − prior logit) by layer, per linguistic category — {MODEL}, known facts',
                 'fig2_lens_curves.png', 'mean δ = logit(context) − logit(prior)',
                 [DELTA_T, LENS_T, TERMS, 'Each line is the mean over the rows of that group; group sizes are in each panel title. '
                  'The mirror shape is expected: rows are grouped by their final answer, so blue must end above 0 and red below 0.', DATA_T])
plt.show()
"""))

cells.append(md("""## 3. What the assertion itself does: shift against the prompt without the assertion

Same panels, but y = $\\delta_{\\text{with assertion}} - \\delta_{\\text{without assertion}}$. This removes what the model would do on the bare question and shows only the push of the assertion. Above 0 = the assertion pushes toward the context answer at that layer."""))
cells.append(code("""FIG3 = lens_grid(LL - LL0, JL - JL0,
                 f'Shift caused by the assertion (δ with − δ without), per linguistic category — {MODEL}, known facts',
                 'fig3_assertion_shift.png', 'mean shift of δ (logits)',
                 ['Shift of δ = δ for the prompt with the assertion minus δ for the same question without the assertion. '
                  'It removes what the model already believes and shows only the push of the assertion: '
                  'above 0 = the assertion moved this layer toward the context answer; 0 = no effect.', DELTA_T, LENS_T, TERMS, DATA_T])
plt.show()
"""))

cells.append(md("""## 4. Exactly which layers: layer-by-layer heatmap (J-lens)

Rows = linguistic categories, columns = layers, colour = mean shift caused by the assertion (J-lens). Left: rows where context won; right: rows where prior won. Same colour scale for both."""))
cells.append(code("""def cat_layer_matrix(Y, won):
    M = []
    for dim, cat in CATS:
        s = (df.dimension == dim).values & (df.category == cat).values & (is_ctx == won)
        M.append(Y[s].mean(0))
    return np.array(M)

S = JL - JL0
Mc, Mp = cat_layer_matrix(S, True), cat_layer_matrix(S, False)
vmax = np.ceil(max(Mc.max(), Mp.max()))
cmap = mpl.colors.LinearSegmentedColormap.from_list('seq_blue', ['#fcfcfb', '#cfe0f6', '#7aaee8', '#2a78d6', '#0d3f80'])
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
for ax, M, ttl in [(axes[0], Mc, 'Context won (final answer = context answer)'), (axes[1], Mp, 'Prior won (final answer = memorised prior answer)')]:
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(X_JL)), X_JL, fontsize=7); ax.set_xlabel('layer'); ax.grid(False)
    ax.set_title(f'{ttl}\\nshift of δ caused by the assertion, J-lens', loc='left')
    for i in range(M.shape[0]):
        for j in range(BAND[0], M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=6,
                    color='white' if M[i, j] > vmax * 0.55 else INK)
axes[0].set_yticks(range(len(CATS)), [f'{c} [{d}]' for d, c in CATS])
fig.colorbar(im, ax=axes, fraction=0.02, label='mean shift of δ (logits)')
caption(fig, 'Each cell = mean shift of δ (δ with the assertion minus δ without it) for one category (row) at one layer (column), read with the J-lens. '
        f'Numbers are printed for layers {BAND[0]}-{X_JL[-1]}; in layers 0-{BAND[0]-1} the shift is about 0. Darker = the assertion pushes harder toward the context answer. Both panels use the same colour scale.',
        DELTA_T, LENS_T, TERMS, DATA_T, y=0.0)
fig.savefig(OUT / 'fig4_heatmap_jlens.png', dpi=150, bbox_inches='tight'); FIG4 = fig; plt.show()
"""))

cells.append(md("""## 5. Where the change happens: distribution of the key layers

Pooled over all categories. Flip layer is shown for rows where context won (for prior-won rows the curve mostly never turns positive). Peak prior layer is shown for prior-won rows."""))
cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
bins = np.arange(-0.5, NL + 0.5, 1)
panels = [
    ('Flip layer\\n(rows where context won)', 'flip_logit_lens', 'flip_jlens', is_ctx),
    ('Divergence layer\\n(all rows)', 'divergence_ll', 'divergence_jl', np.ones(len(df), bool)),
    ('Peak prior layer\\n(rows where prior won)', 'peak_prior_layer_ll', 'peak_prior_layer_jl', ~is_ctx),
]
for ax, (ttl, cl, cj, m) in zip(axes, panels):
    a, b = df.loc[m, cl].dropna(), df.loc[m, cj].dropna()
    ax.hist(a, bins=bins, color=INK2, alpha=0.35, label=f'logit lens (n={len(a)})')
    ax.hist(b, bins=bins, histtype='step', color=CTX, lw=2, label=f'J-lens (n={len(b)})')
    ax.set_title(ttl, loc='left'); ax.set_xlabel('layer'); ax.set_xticks(range(0, NL, 2)); ax.legend(fontsize=8)
axes[0].set_ylabel('number of rows')
fig.tight_layout()
caption(fig, 'Histograms of three layer measures, pooled over all 19 categories. Grey bars: logit lens; blue outline: J-lens.',
        'Flip layer = first layer from which δ stays above 0 up to the last layer (the model leans to the context answer from here on). '
        'Divergence layer = first layer from which δ with the assertion stays more than 2 logits above δ without the assertion (the assertion starts to act). '
        'Peak prior layer = layer where δ is lowest (the memorised answer is strongest). Rows where a measure never happens are not counted.',
        DELTA_T, LENS_T, TERMS, DATA_T)
fig.savefig(OUT / 'fig5_layer_distributions.png', dpi=150); FIG5 = fig; plt.show()

for ttl, cl, cj, m in panels:
    for c in (cl, cj):
        v = df.loc[m, c].dropna()
        print(f'{c:22s} n={len(v):5d}  median={v.median():4.1f}  IQR={v.quantile(.25):.0f}-{v.quantile(.75):.0f}  missing={df.loc[m, c].isna().mean():.1%}')
"""))

cells.append(md("""## 6. Key layers per linguistic category (mean ± 95% CI)

Small differences between categories show up only in the means, because all categories turn within a few layers of each other."""))
cells.append(code("""def mean_ci(v):
    v = v.dropna(); return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0

metrics = [('Flip layer, J-lens\\n(rows where context won)', 'flip_jlens', True),
           ('Divergence layer, J-lens\\n(all rows)', 'divergence_jl', None),
           ('Peak prior layer, J-lens\\n(rows where prior won)', 'peak_prior_layer_jl', False)]
order = beh[['dimension', 'category']].itertuples(index=False, name=None)  # sorted like fig. 1
order = list(order)
fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True)
y = np.arange(len(order))
for ax, (ttl, col, won) in zip(axes, metrics):
    for i, (dim, cat) in enumerate(order):
        m = (df.dimension == dim) & (df.category == cat)
        if won is not None: m &= (df.choice == 'context') == won
        mu, ci = mean_ci(df.loc[m, col])
        ax.errorbar(mu, i, xerr=ci, fmt='o', ms=6, color=CTX if won is not False else PRI, ecolor=MUTED, capsize=2)
    ax.set_title(ttl, loc='left'); ax.set_xlabel('layer'); ax.grid(axis='y', visible=False)
axes[0].set_yticks(y, [f'{c} [{d}]' for d, c in order])
fig.suptitle('Key layers per linguistic category (same order as fig. 1: within each dimension, by % context won)', x=0.01, ha='left')
fig.tight_layout()
caption(fig, 'Dot = mean layer over the rows of the category; bar = 95% confidence interval of the mean. Note the narrow x ranges: categories differ by less than one layer.',
        'Flip layer = first layer from which δ stays above 0 to the end. Divergence layer = first layer from which δ with the assertion stays more than 2 logits above δ without it. '
        'Peak prior layer = layer where δ is lowest (memorised answer strongest).', DELTA_T, LENS_T, TERMS, DATA_T)
fig.savefig(OUT / 'fig6_key_layers_by_category.png', dpi=150); FIG6 = fig; plt.show()
"""))

cells.append(md("""## 7. Summary table per category"""))
cells.append(code("""PEAK = int(X_JL[np.argmax(S[is_ctx].mean(0))])   # J-lens layer where the assertion pushes hardest
rows = []
for dim, cat in order:
    m = (df.dimension == dim).values & (df.category == cat).values
    c, p = m & is_ctx, m & ~is_ctx
    rows.append({
        'dimension': dim, 'category': cat, 'n': m.sum(), 'context won %': 100 * c.sum() / m.sum(),
        'flip JL (ctx) mean': df.loc[c, 'flip_jlens'].mean(), 'flip LL (ctx) mean': df.loc[c, 'flip_logit_lens'].mean(),
        'divergence JL mean': df.loc[m, 'divergence_jl'].mean(),
        'peak prior JL (prior) mean': df.loc[p, 'peak_prior_layer_jl'].mean(),
        f'shift JL L{PEAK} (ctx)': S[c, list(X_JL).index(PEAK)].mean(), f'shift JL L{PEAK} (prior)': S[p, list(X_JL).index(PEAK)].mean(),
        'δ final (ctx)': LL[c, -1].mean(), 'δ final (prior)': LL[p, -1].mean(),
    })
TAB = pd.DataFrame(rows).round(1)
TAB.to_csv(OUT / 'summary_by_category.csv', index=False)
TAB
"""))

cells.append(code("""# separation layer: first layer where the mean curves of context-won and prior-won rows differ by > 2 logits
def first_sep(Y, x, thr=2.0):
    gap = Y[is_ctx].mean(0) - Y[~is_ctx].mean(0)
    idx = np.where(gap > thr)[0]
    return int(x[idx[0]]) if len(idx) else None
print('layer where context-won and prior-won curves separate by >2 logits: logit lens', first_sep(LL, X_LL), '| J-lens', first_sep(JL, X_JL))
print(f'mean assertion shift, layers 0-{BAND[0]-1} (all rows): logit lens {(LL - LL0)[:, :BAND[0]].mean():.2f}, J-lens {S[:, :BAND[0]].mean():.2f}')
print('turning window', BAND, '| peak shift layer (J-lens)', PEAK)
for l in sorted(set(range(BAND[0] - 1, X_JL[-1] + 1)) & set(X_JL)):
    j = list(X_JL).index(l)
    print(f'layer {l:2d}: J-lens shift context-won {S[is_ctx, j].mean():5.1f} | prior-won {S[~is_ctx, j].mean():5.1f}')
"""))

FINDINGS = r"""## Findings

FINDINGS_TEXT
"""
cells.append(md(FINDINGS))

cells.append(code("""# PDF report: text page + all figures + summary table
from textwrap import wrap
def text_page(pdf, title, paragraphs):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.07, 0.95, title, fontsize=15, weight='bold', va='top')
    yy = 0.91
    for p in paragraphs:
        for k, line in enumerate(wrap(p, 95) or ['']):
            fig.text(0.07, yy, line, fontsize=9.5, va='top'); yy -= 0.018
        yy -= 0.012
    pdf.savefig(fig); plt.close(fig)

with PdfPages(RES / f'{MODEL}_lens_report.pdf') as pdf:
    text_page(pdf, f'{MODEL}: where the model changes its mind', REPORT_TEXT)
    for f in [FIG1, FIG2, FIG3, FIG4, FIG5, FIG6]:
        pdf.savefig(f, bbox_inches='tight')
    fig, ax = plt.subplots(figsize=(16, 7)); ax.axis('off')
    t = ax.table(cellText=TAB.values, colLabels=['\\n'.join(wrap(c, 14)) for c in TAB.columns], loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        if r == 0: cell.set_height(cell.get_height() * 2.2)
    ax.set_title('Summary per category (layers: means; shift and δ: logits)', loc='left')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
print('saved', (RES / f'{MODEL}_lens_report.pdf').relative_to(ROOT))
"""))

for c in cells:
    if c.cell_type == 'code':
        c.source = c.source.replace('MODEL_X', MODEL).replace('SUBSET_X', SUBSET).replace('RESDIR_X', RESDIR)
nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
TXT = open(sys.argv[2]).read() if len(sys.argv) > 2 else 'To be written after the first run.'
nb.cells[-2].source = FINDINGS.replace('FINDINGS_TEXT', TXT)
paras = [p.replace('**', '').replace('$', '').lstrip('- ') for p in TXT.split('\n') if p.strip() and not p.startswith('#')]
nb.cells[-1].source = 'REPORT_TEXT = ' + repr(paras) + '\n' + nb.cells[-1].source
NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(__import__('pathlib').Path(NB).parent)}}).execute()
nbf.write(nb, NB)
print('ok', NB)
