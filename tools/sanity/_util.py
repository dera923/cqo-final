import yaml, pandas as pd, numpy as np, os, warnings

SPEC="docs/requirements/core_spec.yml"

def load_spec():
    with open(SPEC, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _csv_path(sp):
    dc = (sp.get("data_contract") or {})
    return dc.get("csv_path", "data/data_estimates.csv")

def _default_columns():
    # columns が無いときの安全デフォルト
    return {
        "y":   {"candidates": ["y","profit","outcome","target","revenue"]},
        "t":   {"candidates": ["t","T","treat","treatment","assigned","action"]},
        "e":   {"candidates": ["e","ps","propensity","pscore","p_hat"]},
        "mu0": {"candidates": ["mu0","m0","y0_hat","mu_0","hat_y0","pred0","predict0","out0"]},
        "mu1": {"candidates": ["mu1","m1","y1_hat","mu_1","hat_y1","pred1","predict1","out1"]},
    }

def _resolve_col(df, entry):
    if isinstance(entry, str):
        return df[entry].to_numpy() if entry in df.columns else None
    if isinstance(entry, dict) and "candidates" in entry:
        for c in entry["candidates"]:
            if c in df.columns: return df[c].to_numpy()
    return None

def _get_colmap(sp):
    dc = (sp.get("data_contract") or {})
    colmap = dc.get("columns")
    return colmap if isinstance(colmap, dict) else _default_columns()

def load_df_and_weight():
    sp = load_spec()
    df = pd.read_csv(_csv_path(sp))
    colmap = _get_colmap(sp)

    y   = _resolve_col(df, colmap.get("y","y"))
    t   = _resolve_col(df, colmap.get("t","t"))
    e   = _resolve_col(df, colmap.get("e","e"))
    mu0 = _resolve_col(df, colmap.get("mu0","mu0"))
    mu1 = _resolve_col(df, colmap.get("mu1","mu1"))

    if t is None:
        tf = (sp.get("t_fallback") or {})
        if tf.get("allow_if_missing", False):
            rng = np.random.default_rng(int(tf.get("seed", 0)))
            t = (rng.uniform(size=len(df)) < (e.astype(float))).astype(int)
            os.makedirs("reports/logs", exist_ok=True)
            open("reports/logs/t_fallback.txt","w").write("bernoulli_from_e\n")
        else:
            raise KeyError("t not found and t_fallback not allowed")

    if (mu0 is None) or (mu1 is None):
        mf = (sp.get("mu_fallback") or {})
        if mf.get("allow_if_missing", False):
            m = float(np.mean(y))
            mu0 = np.full_like(y, m, dtype=float)
            mu1 = np.full_like(y, m, dtype=float)
            warnings.warn("mu_fallback: global_mean was used", RuntimeWarning)
            os.makedirs("reports/logs", exist_ok=True)
            open("reports/logs/mu_fallback.txt","w").write("global_mean\n")
        else:
            raise KeyError("mu0/mu1 not found and mu_fallback not allowed")

    wconf = sp.get("weight", {})
    cand = wconf.get("candidates", ["weight","w","ipw","sample_weight","sw"])
    w = None
    for c in cand:
        if c in df.columns: w = df[c].to_numpy(); break
    if w is None and wconf.get("auto_if_missing", True):
        lo, hi = wconf.get("clip", [1e-6, 1-1e-6])
        e2 = np.clip(e.astype(float), lo, hi)
        p = float(np.mean(t))
        w = t * (p / e2) + (1 - t) * ((1 - p) / (1 - e2))
    if w is None:
        raise KeyError("weight not found and auto_if_missing=false")

    return df, y.astype(float), t.astype(int), e.astype(float), mu0.astype(float), mu1.astype(float), w.astype(float)

def load_df_for_gate():
    sp = load_spec()
    df = pd.read_csv(_csv_path(sp))
    colmap = _get_colmap(sp)

    y = _resolve_col(df, colmap.get("y","y"))
    e = _resolve_col(df, colmap.get("e","e"))
    if y is None or e is None: raise KeyError("y/e not found for gate loader")

    t = _resolve_col(df, colmap.get("t","t"))
    if t is None:
        tf = (sp.get("t_fallback") or {})
        if tf.get("allow_if_missing", False):
            rng = np.random.default_rng(int(tf.get("seed", 0)))
            t = (rng.uniform(size=len(df)) < (e.astype(float))).astype(int)
            os.makedirs("reports/logs", exist_ok=True)
            open("reports/logs/t_fallback.txt","w").write("bernoulli_from_e\n")
        else:
            raise KeyError("t not found and t_fallback not allowed")

    wconf = sp.get("weight", {})
    cand = wconf.get("candidates", ["weight","w","ipw","sample_weight","sw"])
    w = None
    for c in cand:
        if c in df.columns: w = df[c].to_numpy(); break
    if w is None and wconf.get("auto_if_missing", True):
        lo, hi = wconf.get("clip", [1e-6, 1-1e-6])
        e2 = np.clip(e.astype(float), lo, hi)
        p = float(np.mean(t))
        w = t * (p / e2) + (1 - t) * ((1 - p) / (1 - e2))
    if w is None:
        raise KeyError("weight not found and auto_if_missing=false")

    return df, y.astype(float), t.astype(int), e.astype(float), w.astype(float)
