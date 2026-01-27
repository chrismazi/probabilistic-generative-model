#!/usr/bin/env python
"""
Database initialization script.

Creates all tables defined in sql/schema.sql.

Usage:
    python scripts/init_db.py              # Apply schema
    python scripts/init_db.py --dry-run    # Show SQL without executing
    python scripts/init_db.py --drop       # Drop and recreate all tables
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings

console = Console()


def load_schema() -> str:
    """Load SQL schema from file."""
    schema_path = Path(__file__).parent.parent / "sql" / "schema.sql"
    
    if not schema_path.exists():
        console.print(f"[red]Schema file not found: {schema_path}[/red]")
        sys.exit(1)
    
    return schema_path.read_text()


def drop_all_tables(engine) -> None:
    """Drop all tables in the database."""
    console.print("[yellow]Dropping all tables...[/yellow]")
    
    with engine.connect() as conn:
        # Drop views first
        conn.execute(text("DROP VIEW IF EXISTS league_baselines CASCADE"))
        conn.execute(text("DROP VIEW IF EXISTS match_outcomes CASCADE"))
        
        # Drop tables in dependency order
        tables = [
            "coverage_reports",
            "decisions",
            "predictions",
            "model_versions",
            "match_features",
            "scores",
            "matches",
            "teams",
            "leagues",
        ]
        
        for table in tables:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        
        # Drop types
        conn.execute(text("DROP TYPE IF EXISTS decision_outcome CASCADE"))
        conn.execute(text("DROP TYPE IF EXISTS decision_rule CASCADE"))
        conn.execute(text("DROP TYPE IF EXISTS match_status CASCADE"))
        
        # Drop functions
        conn.execute(text("DROP FUNCTION IF EXISTS compute_2h_gt_1h CASCADE"))
        
        conn.commit()
    
    console.print("[green]All tables dropped.[/green]")


def apply_schema(engine, dry_run: bool = False) -> None:
    """Apply the schema to the database."""
    schema = load_schema()
    
    if dry_run:
        console.print(Panel(
            schema[:2000] + "\n... (truncated)",
            title="SQL Schema (Dry Run)",
            expand=False,
        ))
        console.print("[yellow]Dry run - no changes made.[/yellow]")
        return
    
    console.print("[blue]Applying schema...[/blue]")
    
    with engine.connect() as conn:
        # Split by statement and execute
        # Note: This is a simple split; complex schemas might need smarter parsing
        statements = schema.split(";")
        
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and not stmt.startswith("--"):
                try:
                    conn.execute(text(stmt))
                except Exception as e:
                    # Some statements might fail if objects already exist
                    if "already exists" not in str(e).lower():
                        console.print(f"[yellow]Warning: {e}[/yellow]")
        
        conn.commit()
    
    console.print("[green]Schema applied successfully.[/green]")


def verify_tables(engine) -> bool:
    """Verify that core tables exist."""
    required_tables = ["leagues", "teams", "matches", "scores", "predictions"]
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """))
        existing = {row[0] for row in result}
    
    missing = set(required_tables) - existing
    
    if missing:
        console.print(f"[red]Missing tables: {missing}[/red]")
        return False
    
    console.print(f"[green]All required tables present: {required_tables}[/green]")
    return True


def main():
    parser = argparse.ArgumentParser(description="Initialize database schema")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show SQL without executing",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all tables before creating",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify tables exist",
    )
    
    args = parser.parse_args()
    
    console.print(Panel(
        f"Database: {settings.database_url.split('@')[-1] if '@' in settings.database_url else 'localhost'}",
        title="Database Initialization",
    ))
    
    try:
        engine = create_engine(settings.database_url)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        console.print("[green]Database connection OK[/green]")
        
    except Exception as e:
        console.print(f"[red]Database connection failed: {e}[/red]")
        console.print("\n[yellow]Make sure PostgreSQL is running and DATABASE_URL is set correctly.[/yellow]")
        sys.exit(1)
    
    if args.verify:
        success = verify_tables(engine)
        sys.exit(0 if success else 1)
    
    if args.drop:
        if not args.dry_run:
            confirm = input("This will DELETE all data. Type 'yes' to confirm: ")
            if confirm.lower() != "yes":
                console.print("[yellow]Aborted.[/yellow]")
                sys.exit(0)
        drop_all_tables(engine)
    
    apply_schema(engine, dry_run=args.dry_run)
    verify_tables(engine)


if __name__ == "__main__":
    main()
