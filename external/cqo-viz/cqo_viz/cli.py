import click
from pathlib import Path
from .gate import gate_report_main

@click.group()
def cli():
    """cqo-viz - CQO visualization tool."""
    pass

@cli.command(name="gate-report")
@click.option('--csv', required=True, type=click.Path(exists=True),
              help='Input CSV (must include propensity, weight, treatment, features...)')
@click.option('--eps', type=float, default=None, help='Adopted epsilon; if None, compute empirically')
@click.option('--alpha', type=float, default=0.05, help='Tail prob threshold (default 0.05)')
@click.option('--w95-max', type=float, default=10.0, help='Max 95th weight (default 10)')
@click.option('--w99-max', type=float, default=10.0, help='Max 99th weight (default 10)')
@click.option('--smd-max', type=float, default=0.10, help='Max SMD (default 0.10)')
@click.option('--features', required=True, help='Comma-separated feature names for SMD (e.g. X0,X1,X2)')
def gate_report(csv, eps, alpha, w95_max, w99_max, smd_max, features):
    """Compute gates + render PNG and append CSV summary."""
    gate_report_main(csv, eps, alpha, w95_max, w99_max, smd_max, features)

if __name__ == "__main__":
    cli()
