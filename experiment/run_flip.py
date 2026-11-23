"""
Where does the model "change its mind"? Logit lens + J-lens on the EoBench dataset.

Usage (from the repo root; this is how results/qwen3-1.7b/raw/ and results/qwen3-4b/raw/ were made):
    python experiment/run_flip.py --data data/eobench_mcq_with_distractors.csv.gz \
        --model models/qwen3-4b \
        --lens lenses/qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt \
        --limit 11590 --save-every 20 --out results/qwen3-4b/raw/qwen3-4b_raw.parquet

Partial results are written to <out>_parts/ every --save-every rows; rerunning the same
command skips the rows that are already done. All prompts of one fact (the baseline prompt
without the assertion + the 19 linguistic variants) go through the model as one batch
(--batch-size, default 20).

For every prompt and every layer l we compute
    delta_l = logit_l(first token of context_answer) - logit_l(first token of prior_answer)
and the flip layer = the first layer from which delta > 0 holds up to the last layer.
"""
import argparse, json, re, time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RAM_FRACTION = 0.8  # --dtype auto on CPU: use float32 only if the float32 weights take less than this share of RAM


# ---------- model and lens ----------

def pick_dtype(model_path, dtype_name):
    """auto: GPU -> bfloat16; CPU -> float32 if it fits in RAM_FRACTION of RAM, otherwise bfloat16.
    On a CPU without avx512_bf16/AMX, bfloat16 arithmetic is emulated and much slower than float32."""
    if dtype_name != "auto":
        return getattr(torch, dtype_name)
    if torch.cuda.is_available():
        return torch.bfloat16
    import psutil
    idx = Path(model_path) / "model.safetensors.index.json"
    size16 = json.loads(idx.read_text())["metadata"]["total_size"] if idx.exists() else \
        sum(p.stat().st_size for p in Path(model_path).glob("*.safetensors"))
    fits = size16 * 2 < RAM_FRACTION * psutil.virtual_memory().total
    return torch.float32 if fits else torch.bfloat16


def fp32_compute(model):
    """Weights are stored in bfloat16 but computed in float32: before its forward step each decoder
    block gets a temporary float32 copy of its weights; after the step the original bf16 tensors
    are put back.
    Why: on a CPU without avx512_bf16/AMX, bf16 is emulated (3-4x slower), and the whole model
    in float32 does not fit in RAM. The original bf16 weights are never rewritten, so they stay
    memory-mapped from the file and the OS can evict them under memory pressure instead of
    killing the process.
    The result equals a model loaded in float32 (Qwen weights are stored in bf16 anyway).
    Everything outside the blocks (embeddings = tied lm_head, norm, rotary) is converted to float32."""
    for name, mod in model.model.named_children():
        if name != "layers":
            mod.float()
    model.lm_head.float()
    saved = {}

    def up(m, args):
        saved[id(m)] = [(p, p.data) for p in m.parameters()]
        for p, d in saved[id(m)]:
            p.data = d.float()

    def down(m, args, out):
        for p, d in saved.pop(id(m)):
            p.data = d

    for block in model.model.layers:
        block.register_forward_pre_hook(up)
        block.register_forward_hook(down)


def load(model_path, lens_path, dtype_name="auto"):
    tok = AutoTokenizer.from_pretrained(model_path)
    dtype = pick_dtype(model_path, dtype_name)
    # auto on CPU when float32 does not fit: store in bf16, compute in float32 (see fp32_compute)
    upcast = dtype_name == "auto" and dtype == torch.bfloat16 and not torch.cuda.is_available()
    print("dtype:", dtype, "| compute: float32, block by block" if upcast else "")
    # without a GPU, load explicitly on CPU (device_map="auto" tries to offload to disk when RAM is short)
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=device_map,
                                                 low_cpu_mem_usage=True)
    model.eval()
    if upcast:
        fp32_compute(model)
    lens = torch.load(lens_path, weights_only=True)
    # keys can be int or str -> cast to int; convert to float32 once, not for every prompt
    J = {int(k): v.to(model.device).float() for k, v in lens["J"].items()}
    print("lens source_layers:", lens.get("source_layers"), "| n_prompts:", lens.get("n_prompts"),
          "| model layers:", len(model.model.layers))
    return tok, model, J


def build_prompt(tok, text):
    msgs = [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)  # otherwise Qwen3 starts with <think>


def first_token_ids(tok, answer):
    """First token of the answer, in four spellings: with and without a leading space, as written
    and capitalised (at the start of an answer the model usually writes "Red", not "red";
    these are different tokens)."""
    ids = set()
    cap = answer[:1].upper() + answer[1:]
    for v in (answer, " " + answer, cap, " " + cap):
        t = tok.encode(v, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


# ---------- forward pass ----------

@torch.no_grad()
def layer_logits(tok, model, J, prompt, sel=None):
    """Returns (logit_lens [L, V], jlens {layer: [V]}, n_tokens) at the last position.
    sel = token ids the J-lens actually needs: if given, the J-lens is computed only for these
    rows of lm_head (all others = -inf); delta over them is the same as over the full vocabulary.
    The logit lens is always computed over the full vocabulary (needed for ranks and p_choice)."""
    captured = {}
    hook = model.model.layers[-1].register_forward_hook(
        lambda m, i, o: captured.__setitem__("final", o[0] if isinstance(o, tuple) else o))
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    # logits_to_keep=1: the model's own final logits are not used; do not compute them for every position
    out = model(**ids, output_hidden_states=True, logits_to_keep=1)
    hook.remove()

    # hidden_states[0] = embeddings; in HF, hidden_states[-1] is already AFTER the final norm,
    # so the last layer is taken from the hook (before the norm)
    hs = [h[0, -1] for h in out.hidden_states[1:-1]] + [captured["final"][0, -1]]
    hs = torch.stack(hs)                                            # [L, d], output of block l
    ll, jl = lens_logits(model, J, hs, sel)
    return ll, jl, ids.input_ids.shape[1]


@torch.no_grad()
def layer_logits_batch(tok, model, J, prompts, sel=None):
    """Same as layer_logits, but for several prompts in one forward pass
    (left padding; RoPE depends only on relative positions, so the shift caused by padding
    changes nothing). On CPU this is several times faster than one prompt at a time.
    Returns a list of (logit_lens, jlens, n_tokens) in the order of prompts."""
    captured = {}
    hook = model.model.layers[-1].register_forward_hook(
        lambda m, i, o: captured.__setitem__("final", o[0] if isinstance(o, tuple) else o))
    enc = tok(prompts, return_tensors="pt", add_special_tokens=False,
              padding=True, padding_side="left").to(model.device)
    out = model(**enc, output_hidden_states=True, logits_to_keep=1)
    hook.remove()
    hs = torch.stack([h[:, -1] for h in out.hidden_states[1:-1]] + [captured["final"][:, -1]], 1)  # [B, L, d]
    n_tok = enc.attention_mask.sum(1).tolist()
    return [(*lens_logits(model, J, hs[b], sel), n_tok[b]) for b in range(len(prompts))]


@torch.no_grad()
def lens_logits(model, J, hs, sel=None):
    """hs [L, d] = block outputs at the last position -> (logit_lens [L, V], jlens {layer: [V]})."""
    unembed = lambda x: model.lm_head(model.model.norm(x)).float()

    ll = unembed(hs)                                                # logit lens
    layers = [l for l in sorted(J) if l < len(hs)]
    if not layers:
        return ll, {}
    # J-lens: unembed(h @ J^T); the product is taken in float32, as the lens description recommends
    hj = torch.stack([(hs[l].float() @ J[l].T).to(hs.dtype) for l in layers])   # [Lj, d]
    hj = model.model.norm(hj)
    if sel is None:
        lj = model.lm_head(hj).float()                                          # [Lj, V]
    else:
        sel_t = torch.tensor(sel, device=hj.device)
        part = (hj @ model.lm_head.weight[sel_t].T).float()                     # [Lj, |sel|]
        if getattr(model.lm_head, "bias", None) is not None:
            part = part + model.lm_head.bias[sel_t].float()
        lj = torch.full((len(layers), ll.shape[-1]), float("-inf"), device=hj.device)
        lj[:, sel_t] = part
    jl = {l: lj[i] for i, l in enumerate(layers)}
    return ll, jl


def delta(logits, ids_ctx, ids_pri):
    """logsumexp over the context-answer token ids minus logsumexp over the prior-answer token ids."""
    lse = lambda ids: torch.logsumexp(logits[..., ids], dim=-1)
    return (lse(ids_ctx) - lse(ids_pri)).cpu()


def flip_layer(d):
    """First layer from which d > 0 holds up to the last layer (None if never)."""
    pos = (d > 0).tolist()
    for l in range(len(pos)):
        if all(pos[l:]):
            return l
    return None


def strip_assertion(prompt, assertion):
    """The same prompt without the assertion: used to check whether the model knows the prior answer."""
    return prompt.replace(assertion, "").lstrip()


def ranks(logits, ids):
    """Rank of the best spelling of the answer in the full vocabulary at every layer (0 = top-1)."""
    best = logits[..., ids].max(dim=-1).values                      # [L]
    return (logits > best.unsqueeze(-1)).sum(-1).cpu()


def divergence_layer(d, d0, thr):
    """First layer from which the curve with the assertion stays above the curve without it
    by more than thr logits (towards the context answer) up to the last layer."""
    shift = ((d - d0) > thr).tolist()
    for l in range(len(shift)):
        if all(shift[l:]):
            return l
    return None


def peak_prior(d):
    """Layer where the prior answer is furthest ahead of the context answer (minimum delta), and that value."""
    l = int(torch.argmin(d))
    return l, round(float(d[l]), 3)


def curves(ll, jl, ids):
    """delta context-prior and distractor-prior for both lenses."""
    jl_layers = sorted(jl)
    stack = lambda key: torch.stack([delta(jl[l], ids[key], ids["prior"]) for l in jl_layers]) \
        if jl_layers else torch.tensor([])
    return {
        "ll": delta(ll, ids["context"], ids["prior"]),
        "ll_dtr": delta(ll, ids["distractor"], ids["prior"]),
        "jl": stack("context"),
        "jl_dtr": stack("distractor"),
        "jl_layers": jl_layers,
    }


js = lambda t: json.dumps([round(x, 3) if isinstance(x, float) else x for x in t.tolist()])


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N rows (sorted by fact, then category)")
    ap.add_argument("--n-facts", "--sample-per-category", dest="n_facts", type=int, default=None,
                    help="take N random facts (fact_index) with all their rows, i.e. the same N facts "
                         "in every linguistic category (--sample-per-category is the old name)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
                    help="weight precision: auto = GPU -> bfloat16; CPU -> float32 if it fits in RAM, "
                         "otherwise bfloat16 storage with float32 compute")
    ap.add_argument("--full-vocab-jlens", action="store_true",
                    help="compute the J-lens over the full vocabulary (slower, same delta)")
    ap.add_argument("--out", default="results/flip.parquet")
    ap.add_argument("--save-every", type=int, default=20,
                    help="save partial results every N rows to <out>_parts/; "
                         "on a rerun the ids already saved are skipped")
    ap.add_argument("--sep", default=",", help="column separator of --data: ',' (default), ';' or '\\t'")
    ap.add_argument("--batch-size", type=int, default=20,
                    help="how many prompts of one fact go through the model in one forward pass "
                         "(baseline + 19 variants = 20); 1 = one at a time")
    ap.add_argument("--div-thr", type=float, default=2.0,
                    help="threshold (in logits) for the divergence layer against the baseline curve")
    args = ap.parse_args()

    sep = "\t" if args.sep in ("\\t", "tab") else args.sep
    df = pd.read_csv(args.data, sep=sep)
    df = df.dropna(subset=["prompt"]).loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
    if args.n_facts:
        facts = df["fact_index"].drop_duplicates().sample(
            n=args.n_facts, random_state=args.seed)
        df = df[df["fact_index"].isin(facts)].sort_values(["fact_index", "category"])
        print(f"sample: {len(facts)} facts x {df.category.nunique()} categories = {len(df)} rows")
    # fact by fact: the baseline without the assertion is computed once per fact,
    # and every partial result already covers all categories
    df = df.sort_values(["fact_index", "category"], kind="stable")
    if args.limit:
        df = df.head(args.limit)

    # --- partial results and resume ---
    parts_dir = Path(re.sub(r"\.(parquet|csv)$", "", args.out) + "_parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in sorted(parts_dir.glob("part_*.parquet")):
        done |= set(pd.read_parquet(p, columns=["id"]).id)
    n_parts = len(list(parts_dir.glob("part_*.parquet")))
    if done:
        print(f"resume: {len(done)} rows already in {parts_dir}, skipping them")
    df = df[~df["id"].isin(done)]
    total = len(df)
    print(f"to do: {total} rows", flush=True)

    tok, model, J = load(args.model, args.lens, args.dtype)
    rows = []
    t_start, n_done = time.time(), 0

    def flush_part():
        nonlocal rows, n_parts
        if rows:
            pd.DataFrame(rows).to_parquet(parts_dir / f"part_{n_parts:05d}.parquet")
            n_parts += 1
            rows = []

    for fi, g in df.groupby("fact_index", sort=False):
        t0 = time.time()
        r0 = g.iloc[0]
        ans = {"context": r0["context_answer"], "prior": r0["prior_answer"],
               "distractor": r0["distractor_answer"]}
        ids = {k: first_token_ids(tok, v) for k, v in ans.items()}
        # if two options share a first token, the options cannot be told apart
        if set(ids["context"]) & set(ids["prior"]) or set(ids["context"]) & set(ids["distractor"]) \
                or set(ids["prior"]) & set(ids["distractor"]):
            for rid in g["id"]:
                print(f"skip {rid}: first tokens collide")
            continue
        sel = None if args.full_vocab_jlens else sorted(set().union(*ids.values()))

        # 1) the prompt WITHOUT the assertion (does the model know the fact + baseline curve) and
        # 2) all variants of the fact WITH the assertion, in one batch (--batch-size prompts at a time).
        # Note: the baseline is built from the first variant, so it keeps that variant's option order.
        prompts = [build_prompt(tok, strip_assertion(r0["prompt"], r0["assertion"]))] + \
                  [build_prompt(tok, p) for p in g["prompt"]]
        out = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i:i + args.batch_size]
            out += layer_logits_batch(tok, model, J, chunk, sel=sel) if len(chunk) > 1 \
                else [layer_logits(tok, model, J, chunk[0], sel=sel)]
        sec = round((time.time() - t0) / len(g), 2)

        ll0, jl0, _ = out[0]
        sc0 = {k: torch.logsumexp(ll0[-1][v], 0).item() for k, v in ids.items()}
        knows_prior, c0 = max(sc0, key=sc0.get) == "prior", curves(ll0, jl0, ids)

        for (_, r), (ll, jl, n_tok) in zip(g.iterrows(), out[1:]):
            fin = ll[-1]
            scores = {k: torch.logsumexp(fin[v], 0).item() for k, v in ids.items()}
            choice = max(scores, key=scores.get)        # C / M / O in the EoBench paper
            p_choice = torch.softmax(fin, -1)[ids[choice]].sum().item()   # confidence in the answer

            c = curves(ll, jl, ids)
            jl_layers = c["jl_layers"]
            jl_idx = lambda i: jl_layers[i] if i is not None else None
            fj = flip_layer(c["jl"]) if len(c["jl"]) else None
            pk_ll, pkv_ll = peak_prior(c["ll"])
            pk_jl, pkv_jl = peak_prior(c["jl"]) if len(c["jl"]) else (None, None)
            dv_jl = divergence_layer(c["jl"], c0["jl"], args.div_thr) if len(c["jl"]) else None

            try:
                subj_pop = json.loads(r["fact_json"]).get("subj_pop")
            except Exception:
                subj_pop = None

            rows.append({
                # which prompt
                "id": r["id"], "fact_index": fi, "dimension": r["dimension"], "category": r["category"],
                "subj_pop": subj_pop, "prompt_tokens": n_tok,
                # behaviour
                "knows_prior": knows_prior, "choice": choice, "p_choice": round(p_choice, 4),
                "final_delta": round(float(c["ll"][-1]), 3),
                "n_layers": len(c["ll"]),
                # flip: from which layer the context answer stays ahead of the prior to the end (None = never)
                "flip_logit_lens": flip_layer(c["ll"]),
                "flip_jlens": jl_idx(fj),
                # peak prior: where the model "remembers" its own answer most strongly
                "peak_prior_layer_ll": pk_ll, "peak_prior_value_ll": pkv_ll,
                "peak_prior_layer_jl": jl_idx(pk_jl), "peak_prior_value_jl": pkv_jl,
                # divergence: from which layer the assertion moves the curve away from the baseline
                "divergence_ll": divergence_layer(c["ll"], c0["ll"], args.div_thr),
                "divergence_jl": jl_idx(dv_jl),
                # per-layer curves
                "delta_logit_lens": js(c["ll"]), "delta_jlens": js(c["jl"]),
                "delta0_logit_lens": js(c0["ll"]), "delta0_jlens": js(c0["jl"]),   # without the assertion
                "delta_dtr_logit_lens": js(c["ll_dtr"]), "delta_dtr_jlens": js(c["jl_dtr"]),
                "jlens_layers": json.dumps(jl_layers),
                # vocabulary ranks (0 = top-1): if both tokens rank very low, delta is noise
                "rank_ctx_ll": js(ranks(ll, ids["context"])),
                "rank_pri_ll": js(ranks(ll, ids["prior"])),
                # time per prompt: share of the whole fact's forward pass (including the baseline)
                "sec": sec,
            })
            n_done += 1
            el = time.time() - t_start
            eta = el / n_done * (total - n_done)
            print(f"[{n_done}/{total}] {r['id']}: choice={choice} knows_prior={knows_prior} "
                  f"flip LL={rows[-1]['flip_logit_lens']} JL={rows[-1]['flip_jlens']} "
                  f"peak_prior JL={rows[-1]['peak_prior_layer_jl']} div JL={rows[-1]['divergence_jl']} "
                  f"[{rows[-1]['sec']:.1f}s | elapsed {el/3600:.2f}h | ETA {eta/3600:.1f}h]", flush=True)
            if len(rows) >= args.save_every:
                flush_part()

    flush_part()
    parts = sorted(parts_dir.glob("part_*.parquet"))
    if not parts:
        print("no results")
        return
    res = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    res.to_parquet(args.out) if args.out.endswith(".parquet") else res.to_csv(args.out, index=False)
    print(f"\nsaved {len(res)} rows to {args.out} | this run: {n_done} rows in "
          f"{(time.time() - t_start)/3600:.2f}h | mean {res.sec.mean():.1f}s/row")

    # short summary
    ok = res[res.knows_prior & res.choice.isin(["context", "prior"])]
    if len(ok):
        print("\n% context won per category (only facts the model knows):")
        print(ok.groupby("category").choice.apply(lambda c: round((c == "context").mean() * 100, 1)))
        print("\nMean flip layer (only rows where the model chose the context answer):")
        print(ok[ok.choice == "context"].groupby("category")[["flip_logit_lens", "flip_jlens"]].mean().round(1))
        print("\nPeak prior layer and divergence from the baseline curve (all known facts):")
        print(ok.groupby("category")[["peak_prior_layer_jl", "divergence_jl"]].mean().round(1))


if __name__ == "__main__":
    main()
