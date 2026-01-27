#!/usr/bin/env python
"""
Daily data ingestion script.

Fetches matches from football-data.org and stores them in the database.

Usage:
    python scripts/daily_ingest.py                    # Ingest yesterday + today
    python scripts/daily_ingest.py --leagues PL BL1   # Specific leagues
    python scripts/daily_ingest.py --season 2024      # Full season
    python scripts/daily_ingest.py --coverage         # Show coverage report
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data.client import FootballDataClient
from src.data.ingestion import IngestionPipeline
from src.data.quality import run_quality_checks

console = Console()


def display_ingestion_results(results: dict) -> None:
    """Display ingestion results in a nice table."""
    table = Table(title="Ingestion Results")
    table.add_column("League", style="cyan")
    table.add_column("Fetched", justify="right")
    table.add_column("Scores", justify="right")
    table.add_column("Errors", justify="right", style="red")
    
    for league, stats in results.items():
        if "error" in stats:
            table.add_row(league, "-", "-", stats["error"][:30])
        else:
            table.add_row(
                league,
                str(stats.get("total_fetched", 0)),
                str(stats.get("scores_added", 0)),
                str(len(stats.get("errors", []))),
            )
    
    console.print(table)


def display_coverage(pipeline: IngestionPipeline, leagues: list[str], season: str) -> None:
    """Display coverage report for leagues."""
    table = Table(title=f"Coverage Report - Season {season}")
    table.add_column("League", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("With HT", justify="right")
    table.add_column("Coverage %", justify="right")
    table.add_column("Reliable", justify="center")
    
    for league in leagues:
        try:
            cov = pipeline.compute_coverage(league, season)
            reliable = "✓" if cov.is_reliable else "✗"
            style = "green" if cov.is_reliable else "red"
            table.add_row(
                league,
                str(cov.total_matches),
                str(cov.matches_with_ht),
                f"{cov.ht_coverage_pct:.1f}%",
                f"[{style}]{reliable}[/{style}]",
            )
        except Exception as e:
            table.add_row(league, "-", "-", "-", f"[red]Error: {str(e)[:20]}[/red]")
    
    console.print(table)


def display_quality_report() -> None:
    """Run and display quality checks."""
    console.print("[blue]Running quality checks...[/blue]")
    
    report = run_quality_checks()
    
    status = "[green]HEALTHY[/green]" if report.is_healthy else "[red]ISSUES FOUND[/red]"
    
    console.print(Panel(
        f"Status: {status}\n"
        f"Total matches: {report.total_matches}\n"
        f"Total scores: {report.total_scores}\n"
        f"Errors: {report.error_count}\n"
        f"Warnings: {report.warning_count}",
        title="Data Quality Report",
    ))
    
    if report.issues:
        table = Table(title="Issues")
        table.add_column("Type", style="cyan")
        table.add_column("Severity")
        table.add_column("Description")
        
        for issue in report.issues[:20]:  # Limit display
            sev_style = "red" if issue.severity == "error" else "yellow"
            table.add_row(
                issue.issue_type,
                f"[{sev_style}]{issue.severity}[/{sev_style}]",
                issue.description,
            )
        
        console.print(table)
        
        if len(report.issues) > 20:
            console.print(f"[dim]... and {len(report.issues) - 20} more issues[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Daily data ingestion")
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=None,
        help="League codes to ingest (default: all supported)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Ingest full season (e.g., 2024 for 2024-25)",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Show coverage report only",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Run quality checks after ingestion",
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not settings.football_data_api_key:
        console.print("[red]Error: FOOTBALL_DATA_API_KEY not set[/red]")
        console.print("Set it in .env or as an environment variable")
        sys.exit(1)
    
    # Initialize
    client = FootballDataClient()
    pipeline = IngestionPipeline(client=client)
    
    leagues = args.leagues or list(client.SUPPORTED_LEAGUES.keys())
    
    console.print(Panel(
        f"Leagues: {', '.join(leagues)}",
        title="Data Ingestion",
    ))
    
    # Coverage report only
    if args.coverage:
        season = f"{args.season or 2024}-{str((args.season or 2024) + 1)[-2:]}"
        display_coverage(pipeline, leagues, season)
        return
    
    # Determine date range
    if args.season:
        # Full season
        console.print(f"[blue]Ingesting full season {args.season}...[/blue]")
        
        results = {}
        for league in leagues:
            try:
                stats = pipeline.ingest_matches(
                    league_code=league,
                    season=args.season,
                )
                results[league] = stats
                console.print(f"  {league}: {stats['total_fetched']} matches")
            except Exception as e:
                results[league] = {"error": str(e)}
                console.print(f"  [red]{league}: Error - {e}[/red]")
    
    elif args.date_from or args.date_to:
        # Custom date range
        date_from = date.fromisoformat(args.date_from) if args.date_from else None
        date_to = date.fromisoformat(args.date_to) if args.date_to else None
        
        console.print(f"[blue]Ingesting {date_from} to {date_to}...[/blue]")
        
        results = {}
        for league in leagues:
            try:
                stats = pipeline.ingest_matches(
                    league_code=league,
                    date_from=date_from,
                    date_to=date_to,
                )
                results[league] = stats
            except Exception as e:
                results[league] = {"error": str(e)}
    
    else:
        # Default: yesterday + today
        console.print("[blue]Ingesting yesterday and today...[/blue]")
        results = pipeline.ingest_yesterday_and_today(league_codes=leagues)
    
    display_ingestion_results(results)
    
    # Quality checks
    if args.quality:
        display_quality_report()
    
    console.print("[green]Done![/green]")


if __name__ == "__main__":
    main()
