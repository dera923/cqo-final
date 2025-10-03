from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console

app = typer.Typer(help="CQO 統合コマンド: データのシミュレーションと検証")
console = Console(width=100)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@app.command(help="シミュレーションデータを生成してCSVに保存します")
def simulate(
    n_samples: int = typer.Option(1000, "--n-samples", "-n", help="サンプル数"),
    treatment_effect: float = typer.Option(1.0, "--treatment-effect", "-t", help="処置効果"),
    n_features: int = typer.Option(3, "--n-features", "-f", help="特徴量の次元"),
    seed: Optional[int] = typer.Option(42, "--seed", help="乱数シード（Noneで未固定）"),
    output: Path = typer.Option(Path("data/simulated.csv"), "--output", "-o", help="出力CSVパス"),
    level: str = typer.Option("INFO", "--level", help="ログレベル（DEBUG/INFO/WARNING/ERROR）"),
):
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n_samples, n_features)

    logit_ps = 0.5 * X[:, 0] + 0.5 * X[:, 1] - X[:, 2 % n_features]
    ps = 1.0 / (1.0 + np.exp(-logit_ps))
    T = np.random.binomial(1, ps)

    Y0 = X[:, 0] + np.random.randn(n_samples)
    Y1 = Y0 + treatment_effect * 0.2 + 0.2 * X[:, 2 % n_features]
    Y = T * Y1 + (1 - T) * Y0

    df = pd.DataFrame(
        {
            **{f"x{i}": X[:, i] for i in range(n_features)},
            "t": T,
            "y": Y,
        }
    )

    _ensure_parent(output)
    df.to_csv(output, index=False)
    console.print(f"[bold green]Saved:[/bold green] {output}  (rows={len(df)})")


@app.command(help="CSV を読み込み、基本統計を表示します")
def validate(
    input: Path = typer.Option(Path("data/simulated.csv"), "--input", "-i", help="入力CSV"),
):
    if not input.exists():
        raise typer.BadParameter(f"CSV が見つかりません: {input}")
    df = pd.read_csv(input)
    console.print(f"[bold]Rows[/bold]: {len(df)}, [bold]Cols[/bold]: {len(df.columns)}")
    console.print(df.head())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
