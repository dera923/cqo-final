from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_epsilon_star(propensities: np.ndarray, alpha: float) -> float:
    # symmetric trimming の簡易近似：下側 alpha/2 分位を eps* とする（保守的に 0.01 下限）
    eps = float(np.quantile(propensities, alpha / 2))
    return max(eps, 0.01)


def compute_tail_prob(propensities: np.ndarray, eps: float) -> float:
    p = propensities
    return float(((p < eps) | (p > 1 - eps)).mean())


def compute_smd(df: pd.DataFrame, treatment_col: str, feature_cols: list[str]) -> float:
    tr = df[df[treatment_col] == 1]
    ct = df[df[treatment_col] == 0]
    smds = []
    for f in feature_cols:
        mt, mc = tr[f].mean(), ct[f].mean()
        vt, vc = tr[f].var(), ct[f].var()
        pooled = np.sqrt((vt + vc) / 2) if (vt + vc) > 0 else 1.0
        smds.append(abs(mt - mc) / pooled)
    return float(max(smds) if smds else 0.0)


def render_report_png(metrics: dict, out_png: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    rows = [
        ["metric", "value"],
        ["epsilon_star", f"{metrics['epsilon_star']:.4f}"],
        ["adopted_epsilon", f"{metrics['adopted_epsilon']:.4f}"],
        ["w95", f"{metrics['w95']:.3f}"],
        ["w99", f"{metrics['w99']:.3f}"],
        ["tail_prob", f"{metrics['tail_prob']:.4f}"],
        ["max_SMD", f"{metrics['max_SMD']:.4f}"],
        ["overall", "PASS" if metrics["overall_pass"] else "FAIL"],
    ]
    tab = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.4, 0.4])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.6)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"✓ wrote {out_png}")


def gate_report_main(
    csv_path: str,
    eps: float,
    alpha: float,
    w95_max: float,
    w99_max: float,
    smd_max: float,
    features: str,
):
    df = pd.read_csv(csv_path)
    # 期待される列名
    required = {"propensity", "weight", "treatment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    prop = df["propensity"].to_numpy()
    w = df["weight"].to_numpy()
    # eps*
    eps_star = compute_epsilon_star(prop, alpha)
    eps_adopt = eps if eps is not None else eps_star
    # trimming 後の重み分布
    mask = (prop >= eps_adopt) & (prop <= 1 - eps_adopt)
    w_trim = w[mask]
    w95 = float(np.percentile(w_trim, 95)) if w_trim.size else float("nan")
    w99 = float(np.percentile(w_trim, 99)) if w_trim.size else float("nan")
    # tail prob
    tail = compute_tail_prob(prop, eps_adopt)
    # SMD
    feats = [s.strip() for s in features.split(",") if s.strip()]
    max_smd = compute_smd(df, "treatment", feats)
    # 合否
    ok = (tail <= alpha) and (w95 <= w95_max) and (w99 <= w99_max) and (max_smd <= smd_max)
    metrics = dict(
        epsilon_star=eps_star,
        adopted_epsilon=eps_adopt,
        w95=w95,
        w99=w99,
        tail_prob=tail,
        max_SMD=max_smd,
        overall_pass=ok,
    )
    # CSV 追記＆PNG
    tab_dir = Path("reports/tables")
    fig_dir = Path("reports/figures")
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([metrics])
    sm_path = tab_dir / "summary_metrics.csv"
    if sm_path.exists():
        row.to_csv(sm_path, mode="a", index=False, header=False)
    else:
        row.to_csv(sm_path, index=False)
    print(f"✓ appended {sm_path}")
    render_report_png(metrics, fig_dir / "gate_report.png")
