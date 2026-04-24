"""CLI entrypoint for AsesorFinan."""

from __future__ import annotations

import logging
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from asesorfinan.config import settings
from asesorfinan.models import RiskProfile, UserProfile
from asesorfinan.orchestrator import run_pipeline

_VALID_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}

app = typer.Typer(
    help="AsesorFinan — Asesor financiero multi-agente con ML",
    no_args_is_help=True,
    invoke_without_command=False,
)
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def analyze(
    capital: float = typer.Option(..., "--capital", "-c", help="Capital disponible en USD"),
    horizon: int = typer.Option(..., "--horizon", "-h", help="Horizonte de inversión en meses"),
    risk: RiskProfile = typer.Option(..., "--risk", "-r", help="Perfil de riesgo: conservador | moderado | agresivo"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Intervalo yfinance: 1m 2m 5m 15m 30m 90m 1h 1d 5d 1wk"),
    exclude: Optional[list[str]] = typer.Option(None, "--exclude", "-e", help="Tickers a excluir"),
    max_positions: int = typer.Option(10, "--max-pos", help="Máximo de posiciones en el portafolio"),
    prediction_bars: int = typer.Option(0, "--prediction-bars", help="Override horizonte de predicción en barras (0=auto)"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Genera un análisis de portafolio completo para el perfil dado."""
    _setup_logging(log_level)

    if interval not in _VALID_INTERVALS:
        console.print(f"[bold red]Intervalo inválido:[/bold red] '{interval}'. Opciones: {', '.join(sorted(_VALID_INTERVALS))}")
        raise typer.Exit(1)

    # Apply interval overrides before the pipeline imports any cached settings values
    settings.data_interval = interval
    if prediction_bars > 0:
        settings.prediction_horizon_bars = prediction_bars

    profile = UserProfile(
        capital=capital,
        horizon_months=horizon,
        risk_profile=risk,
        excluded_assets=exclude or [],
        max_positions=max_positions,
    )

    intraday_note = f" ([yellow]intraday {interval}[/yellow])" if settings.is_intraday else ""
    console.print(Panel(
        f"[bold cyan]AsesorFinan[/bold cyan] — iniciando análisis{intraday_note}\n"
        f"Capital: [green]${capital:,.0f}[/green] | "
        f"Horizonte: [yellow]{horizon} meses[/yellow] | "
        f"Perfil: [magenta]{risk.value}[/magenta] | "
        f"Intervalo: [cyan]{interval}[/cyan]",
        expand=False,
    ))

    with console.status("[bold green]Ejecutando pipeline multi-agente...[/bold green]", spinner="dots"):
        state = run_pipeline(profile)

    if state.error:
        console.print(f"[bold red]Error:[/bold red] {state.error}")
        raise typer.Exit(1)

    _print_portfolio(state)
    _print_risk(state)
    _print_narrative(state)


def _print_portfolio(state) -> None:
    portfolio = state.portfolio
    if not portfolio:
        return

    table = Table(title="Portafolio Recomendado", show_lines=True)
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Peso %", justify="right")
    table.add_column("Monto USD", justify="right")
    table.add_column("Cluster", style="dim")
    table.add_column("Ret. esperado", justify="right")

    for a in portfolio.allocations:
        table.add_row(
            a.ticker,
            f"{a.weight*100:.1f}%",
            f"${a.amount_usd:,.0f}",
            a.cluster_label,
            f"{a.predicted_return_pct:+.1f}%",
        )

    console.print(table)
    console.print(
        f"  E[Retorno anual]: [green]{portfolio.expected_annual_return*100:.1f}%[/green]  "
        f"Volatilidad: [yellow]{portfolio.annual_volatility*100:.1f}%[/yellow]  "
        f"Sharpe: [cyan]{portfolio.sharpe_ratio:.2f}[/cyan]\n"
    )


def _print_risk(state) -> None:
    risk = state.risk_report
    if not risk:
        return

    color_map = {"verde": "green", "amarillo": "yellow", "rojo": "red"}
    color = color_map.get(risk.signal.value, "white")
    console.print(Panel(
        risk.notes,
        title=f"[{color}]Análisis de Riesgo — {risk.signal.value.upper()}[/{color}]",
        border_style=color,
    ))


def _print_narrative(state) -> None:
    report = state.advisor_report
    if not report:
        return

    console.print(Panel(
        report.narrative,
        title="[bold]Informe del Asesor[/bold]",
        border_style="blue",
    ))


if __name__ == "__main__":
    app()
