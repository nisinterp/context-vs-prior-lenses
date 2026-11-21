"""
Где модель «меняет мнение»: logit lens + J-lens на датасете EoB.

Запуск (из корня репозитория; так получены results/qwen3-1.7b/raw/ и results/qwen3-4b/raw/):
    python experiment/run_flip.py --data data/eobench_mcq_with_distractors.csv.gz --sep , \
        --model models/qwen3-4b \
        --lens lenses/qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt \
        --limit 11590 --save-every 20 --out results/qwen3-4b/raw/qwen3-4b_raw.parquet

Частичные результаты пишутся в <out>_parts/ каждые --save-every строк; при перезапуске
той же командой уже посчитанные строки пропускаются. Все промпты одного факта
(базовый + 19 вариантов) идут одним батчем (--batch-size, по умолчанию 20).

Для каждого примера считаем по слоям
    delta_l = logit_l(первый токен context_answer) - logit_l(первый токен prior_answer)
и слой перелома = первый слой, начиная с которого delta > 0 держится до конца.
"""
import argparse, json, re, time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------- модель и линза ----------

def pick_dtype(model_path, dtype_name):
    """auto: GPU → bfloat16; CPU → float32, если влезает в ~80% RAM, иначе bfloat16.
    На CPU без avx512_bf16/AMX bfloat16 эмулируется и примерно вдвое медленнее float32."""
    if dtype_name != "auto":
        return getattr(torch, dtype_name)
    if torch.cuda.is_available():
        return torch.bfloat16
    import psutil
    idx = Path(model_path) / "model.safetensors.index.json"
    size16 = json.loads(idx.read_text())["metadata"]["total_size"] if idx.exists() else \
        sum(p.stat().st_size for p in Path(model_path).glob("*.safetensors"))
    fits = size16 * 2 < 0.8 * psutil.virtual_memory().total
    return torch.float32 if fits else torch.bfloat16


def fp32_compute(model):
    """Веса храним в bfloat16, а считаем в float32: каждый блок перед своим шагом получает
    временную float32-копию весов, после шага возвращаются исходные bf16-тензоры.
    Зачем: на CPU без avx512_bf16/AMX bf16 эмулируется (в 3-4 раза медленнее), а целиком
    в float32 модель не влезает в RAM. Исходные bf16-веса не переписываются, поэтому остаются
    отображёнными из файла и ОС может вытеснить их при нехватке памяти, а не убить процесс.
    Результат тот же, что у модели, загруженной в float32 (веса Qwen и так хранятся в bf16).
    Всё вне блоков (эмбеддинги = lm_head, norm, rotary) целиком переводим в float32."""
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
    # auto на CPU, если float32 не влезает: хранение bf16, счёт float32 (см. fp32_compute)
    upcast = dtype_name == "auto" and dtype == torch.bfloat16 and not torch.cuda.is_available()
    print("dtype:", dtype, "| compute: float32, block by block" if upcast else "")
    # без GPU — явно на CPU (device_map="auto" при нехватке RAM пытается выгружать на диск)
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=device_map,
                                                 low_cpu_mem_usage=True)
    model.eval()
    if upcast:
        fp32_compute(model)
    lens = torch.load(lens_path, weights_only=True)
    # ключи бывают int или str — приводим к int; в float32 один раз, а не на каждом примере
    J = {int(k): v.to(model.device).float() for k, v in lens["J"].items()}
    print("lens source_layers:", lens.get("source_layers"), "| n_prompts:", lens.get("n_prompts"),
          "| model layers:", len(model.model.layers))
    return tok, model, J


def build_prompt(tok, text):
    msgs = [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)  # у Qwen3 иначе первым будет <think>


def first_token_ids(tok, answer):
    """Первый токен ответа — без пробела / с пробелом, как написано и с заглавной буквы
    (в начале ответа модель обычно пишет «Red», а не «red» — это разные токены)."""
    ids = set()
    cap = answer[:1].upper() + answer[1:]
    for v in (answer, " " + answer, cap, " " + cap):
        t = tok.encode(v, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


# ---------- прогон ----------

@torch.no_grad()
def layer_logits(tok, model, J, prompt, sel=None):
    """Возвращает (logit_lens [L, V], jlens {layer: [V]}) на последней позиции.
    sel — id токенов, которые реально нужны J-линзе: если задан, J-lens считается только
    по этим строкам lm_head (остальные = -inf); δ по ним получается тот же, что и по всему словарю.
    Logit lens всегда по всему словарю — он нужен для рангов и p_choice."""
    captured = {}
    hook = model.model.layers[-1].register_forward_hook(
        lambda m, i, o: captured.__setitem__("final", o[0] if isinstance(o, tuple) else o))
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    # logits_to_keep=1: финальные логиты модели не используем, не считаем их по всем позициям
    out = model(**ids, output_hidden_states=True, logits_to_keep=1)
    hook.remove()

    # hidden_states[0] — эмбеддинги; hidden_states[-1] в HF уже ПОСЛЕ norm,
    # поэтому последний слой берём из хука (до norm)
    hs = [h[0, -1] for h in out.hidden_states[1:-1]] + [captured["final"][0, -1]]
    hs = torch.stack(hs)                                            # [L, d], выход блока l
    ll, jl = lens_logits(model, J, hs, sel)
    return ll, jl, ids.input_ids.shape[1]


@torch.no_grad()
def layer_logits_batch(tok, model, J, prompts, sel=None):
    """То же, что layer_logits, но для нескольких промптов за один прогон модели
    (left padding; RoPE зависит только от относительных позиций, так что сдвиг из-за паддинга
    ничего не меняет). На CPU это в несколько раз быстрее, чем по одному.
    Возвращает список (logit_lens, jlens, n_tokens) в порядке prompts."""
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
    """hs [L, d] — выходы блоков на последней позиции → (logit_lens [L, V], jlens {layer: [V]})."""
    unembed = lambda x: model.lm_head(model.model.norm(x)).float()

    ll = unembed(hs)                                                # logit lens
    layers = [l for l in sorted(J) if l < len(hs)]
    if not layers:
        return ll, {}
    # J-lens: unembed(h @ J^T); умножаем в float32, как советует описание линзы
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
    lse = lambda ids: torch.logsumexp(logits[..., ids], dim=-1)
    return (lse(ids_ctx) - lse(ids_pri)).cpu()


def flip_layer(d):
    pos = (d > 0).tolist()
    for l in range(len(pos)):
        if all(pos[l:]):
            return l
    return None


def strip_assertion(prompt, assertion):
    """Тот же промпт без утверждения — проверка, знает ли модель prior."""
    return prompt.replace(assertion, "").lstrip()


def ranks(logits, ids):
    """Место лучшего из вариантов токена среди всего словаря на каждом слое (0 = топ-1)."""
    best = logits[..., ids].max(dim=-1).values                      # [L]
    return (logits > best.unsqueeze(-1)).sum(-1).cpu()


def divergence_layer(d, d0, thr):
    """Первый слой, начиная с которого кривая с утверждением сдвинута к контексту
    относительно кривой без утверждения больше чем на thr логитов (и так до конца)."""
    shift = ((d - d0) > thr).tolist()
    for l in range(len(shift)):
        if all(shift[l:]):
            return l
    return None


def peak_prior(d):
    """Слой, где prior сильнее всего впереди контекста (минимум δ), и само значение."""
    l = int(torch.argmin(d))
    return l, round(float(d[l]), 3)


def curves(ll, jl, ids):
    """δ контекст−prior и дистрактор−prior для обеих линз."""
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
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-per-category", type=int, default=None,
                    help="взять N случайных фактов (fact_index) и все их строки — "
                         "одни и те же N фактов в каждой категории")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
                    help="точность весов: auto = GPU→bfloat16, CPU→float32 если влезает в RAM, иначе bfloat16")
    ap.add_argument("--full-vocab-jlens", action="store_true",
                    help="считать J-lens по всему словарю (медленнее, результат δ тот же)")
    ap.add_argument("--out", default="results/flip.parquet")
    ap.add_argument("--save-every", type=int, default=20,
                    help="сохранять частичные результаты каждые N примеров в <out>_parts/; "
                         "при перезапуске уже посчитанные id пропускаются")
    ap.add_argument("--sep", default="\t", help="разделитель столбцов: '\\t' (по умолчанию), ';' или ','")
    ap.add_argument("--batch-size", type=int, default=20,
                    help="сколько промптов одного факта считать за один прогон модели "
                         "(базовый + 19 вариантов = 20); 1 = по одному, как раньше")
    ap.add_argument("--div-thr", type=float, default=2.0,
                    help="порог (в логитах) для слоя расхождения с кривой без утверждения")
    args = ap.parse_args()

    sep = "\t" if args.sep in ("\\t", "tab") else args.sep
    df = pd.read_csv(args.data, sep=sep)
    df = df.dropna(subset=["prompt"]).loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
    if args.sample_per_category:
        facts = df["fact_index"].drop_duplicates().sample(
            n=args.sample_per_category, random_state=args.seed)
        df = df[df["fact_index"].isin(facts)].sort_values(["fact_index", "category"])
        print(f"sample: {len(facts)} facts x {df.category.nunique()} categories = {len(df)} rows")
    # по фактам подряд: базовый прогон без утверждения считается один раз на факт,
    # а частичные результаты сразу покрывают все категории
    df = df.sort_values(["fact_index", "category"], kind="stable")
    if args.limit:
        df = df.head(args.limit)

    # --- частичные результаты и возобновление ---
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
        # если первые токены у вариантов совпадают, по ним ничего не различить
        if set(ids["context"]) & set(ids["prior"]) or set(ids["context"]) & set(ids["distractor"]) \
                or set(ids["prior"]) & set(ids["distractor"]):
            for rid in g["id"]:
                print(f"skip {rid}: first tokens collide")
            continue
        sel = None if args.full_vocab_jlens else sorted(set().union(*ids.values()))

        # 1) прогон БЕЗ утверждения (знает ли модель факт + базовая кривая) и
        # 2) все варианты факта С утверждением — одним батчем (по --batch-size промптов)
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
            choice = max(scores, key=scores.get)        # C / M / O из статьи
            p_choice = torch.softmax(fin, -1)[ids[choice]].sum().item()   # уверенность в ответе

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
                # что за пример
                "id": r["id"], "fact_index": fi, "dimension": r["dimension"], "category": r["category"],
                "subj_pop": subj_pop, "prompt_tokens": n_tok,
                # поведение
                "knows_prior": knows_prior, "choice": choice, "p_choice": round(p_choice, 4),
                "final_delta": round(float(c["ll"][-1]), 3),
                "n_layers": len(c["ll"]),
                # перелом: с какого слоя контекст впереди prior до конца (None = не наступил)
                "flip_logit_lens": flip_layer(c["ll"]),
                "flip_jlens": jl_idx(fj),
                # пик prior: где модель сильнее всего «помнит» свой ответ
                "peak_prior_layer_ll": pk_ll, "peak_prior_value_ll": pkv_ll,
                "peak_prior_layer_jl": jl_idx(pk_jl), "peak_prior_value_jl": pkv_jl,
                # расхождение: с какого слоя утверждение сдвигает кривую относительно базовой
                "divergence_ll": divergence_layer(c["ll"], c0["ll"], args.div_thr),
                "divergence_jl": jl_idx(dv_jl),
                # кривые по слоям
                "delta_logit_lens": js(c["ll"]), "delta_jlens": js(c["jl"]),
                "delta0_logit_lens": js(c0["ll"]), "delta0_jlens": js(c0["jl"]),   # без утверждения
                "delta_dtr_logit_lens": js(c["ll_dtr"]), "delta_dtr_jlens": js(c["jl_dtr"]),
                "jlens_layers": json.dumps(jl_layers),
                # ранги в словаре (0 = топ-1): если оба токена глубоко, δ — шум
                "rank_ctx_ll": js(ranks(ll, ids["context"])),
                "rank_pri_ll": js(ranks(ll, ids["prior"])),
                # время на пример: доля прогона всего факта (вместе с базовым)
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

    # краткая сводка
    ok = res[res.knows_prior & res.choice.isin(["context", "prior"])]
    if len(ok):
        print("\nCFR по категориям (только факты, которые модель знает):")
        print(ok.groupby("category").choice.apply(lambda c: round((c == "context").mean() * 100, 1)))
        print("\nСредний слой перелома (только где модель выбрала контекст):")
        print(ok[ok.choice == "context"].groupby("category")[["flip_logit_lens", "flip_jlens"]].mean().round(1))
        print("\nПик prior и расхождение с базовой кривой (все знакомые факты):")
        print(ok.groupby("category")[["peak_prior_layer_jl", "divergence_jl"]].mean().round(1))


if __name__ == "__main__":
    main()
