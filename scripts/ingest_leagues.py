"""
Multi-League Data Ingestion Script.

Ingests match data from multiple top football leagues:
- PL: Premier League (England)
- BL1: Bundesliga (Germany)
- SA: Serie A (Italy)
- PD: La Liga (Spain)
- FL1: Ligue 1 (France)
- ELC: Championship (England)
- DED: Eredivisie (Netherlands)
- PPL: Primeira Liga (Portugal)

Usage:
    python scripts/ingest_leagues.py                    # Ingest default (top 5)
    python scripts/ingest_leagues.py --leagues PL BL1   # Specific leagues
    python scripts/ingest_leagues.py --season 2024      # Specific season
    python scripts/ingest_leagues.py --all              # All 8 leagues
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.data.client import FootballDataClient
from src.db.connection import get_session


# =============================================================================
# Configuration
# =============================================================================

# Available leagues with their full names and countries
AVAILABLE_LEAGUES = {
    "PL": {"name": "Premier League", "country": "England"},
    "BL1": {"name": "Bundesliga", "country": "Germany"},
    "SA": {"name": "Serie A", "country": "Italy"},
    "PD": {"name": "La Liga", "country": "Spain"},
    "FL1": {"name": "Ligue 1", "country": "France"},
    "ELC": {"name": "Championship", "country": "England"},
    "DED": {"name": "Eredivisie", "country": "Netherlands"},
    "PPL": {"name": "Primeira Liga", "country": "Portugal"},
}

# Default leagues to ingest (top 5 European leagues)
DEFAULT_LEAGUES = ["PL", "BL1", "SA", "PD", "FL1"]


def ensure_league_exists(session, code: str, name: str, country: str) -> int:
    """Ensure league exists in DB, return its ID."""
    try:
        result = session.execute(
            text("SELECT id FROM leagues WHERE code = :code"), 
            {"code": code}
        ).fetchone()
        
        if result:
            return result[0]
        
        # Insert new league
        session.execute(
            text("""
                INSERT INTO leagues (code, name, country, created_at)
                VALUES (:code, :name, :country, NOW())
            """),
            {"code": code, "name": name, "country": country}
        )
        session.commit()
        
        result = session.execute(
            text("SELECT id FROM leagues WHERE code = :code"),
            {"code": code}
        ).fetchone()
        league_id = result[0]
        print(f"  ✓ Created league: {name} ({code})")
        return league_id
    except Exception as e:
        session.rollback()
        raise e


def ensure_team_exists(session, api_team, league_id: int, team_cache: dict) -> int:
    """Ensure team exists in DB, return its ID.
    
    Uses 'external_id' column which stores the API's team ID.
    """
    # Check cache first
    if api_team.id in team_cache:
        return team_cache[api_team.id]
    
    try:
        # Check if team exists by external_id (API ID)
        result = session.execute(
            text("SELECT id FROM teams WHERE external_id = :external_id"),
            {"external_id": api_team.id}
        ).fetchone()
        
        if result:
            team_cache[api_team.id] = result[0]
            return result[0]
        
        # Insert new team
        short_name = api_team.short_name if api_team.short_name else api_team.name[:3].upper()
        tla = api_team.tla if hasattr(api_team, 'tla') and api_team.tla else short_name[:3].upper()
        
        session.execute(
            text("""
                INSERT INTO teams (external_id, name, short_name, tla, created_at)
                VALUES (:external_id, :name, :short_name, :tla, NOW())
            """),
            {
                "external_id": api_team.id, 
                "name": api_team.name, 
                "short_name": short_name, 
                "tla": tla
            }
        )
        session.commit()
        
        result = session.execute(
            text("SELECT id FROM teams WHERE external_id = :external_id"),
            {"external_id": api_team.id}
        ).fetchone()
        team_id = result[0]
        team_cache[api_team.id] = team_id
        return team_id
    except Exception as e:
        session.rollback()
        raise e


def ingest_match(session, match, league_id: int, home_team_id: int, away_team_id: int) -> bool:
    """Ingest a single match, return True if inserted (new).
    
    Uses 'external_id' column which stores the API's match ID.
    """
    try:
        # Check if match exists
        result = session.execute(
            text("SELECT id FROM matches WHERE external_id = :external_id"),
            {"external_id": match.id}
        ).fetchone()
        
        match_status = match.status.value if match.status else 'SCHEDULED'
        
        if result:
            match_id = result[0]
            # Update match status if changed
            session.execute(
                text("""
                    UPDATE matches SET status = :status, matchday = :matchday, updated_at = NOW()
                    WHERE id = :id
                """),
                {"status": match_status, "matchday": match.matchday, "id": match_id}
            )
            is_new = False
        else:
            # Insert new match - use 'season' column as VARCHAR
            season_str = str(match.season.get('id')) if match.season and match.season.get('id') else None
            session.execute(
                text("""
                    INSERT INTO matches (external_id, league_id, home_team_id, away_team_id, 
                                         kickoff_utc, matchday, status, season, created_at)
                    VALUES (:external_id, :league_id, :home_team_id, :away_team_id,
                            :kickoff_utc, :matchday, :status, :season, NOW())
                """),
                {
                    "external_id": match.id,
                    "league_id": league_id,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "kickoff_utc": match.utc_date,
                    "matchday": match.matchday,
                    "status": match_status,
                    "season": season_str,
                }
            )
            session.commit()
            
            result = session.execute(
                text("SELECT id FROM matches WHERE external_id = :external_id"),
                {"external_id": match.id}
            ).fetchone()
            match_id = result[0]
            is_new = True
        
        # Insert/update scores if match is finished
        if match_status == 'FINISHED' and match.score:
            ht_home = match.score.half_time.home if match.score.half_time else None
            ht_away = match.score.half_time.away if match.score.half_time else None
            ft_home = match.score.full_time.home if match.score.full_time else None
            ft_away = match.score.full_time.away if match.score.full_time else None
            
            if ht_home is not None:
                # Check if scores exist
                result = session.execute(
                    text("SELECT id FROM scores WHERE match_id = :match_id"),
                    {"match_id": match_id}
                ).fetchone()
                
                if result:
                    session.execute(
                        text("""
                            UPDATE scores SET ht_home = :ht_home, ht_away = :ht_away, 
                                              ft_home = :ft_home, ft_away = :ft_away, updated_at = NOW()
                            WHERE match_id = :match_id
                        """),
                        {"ht_home": ht_home, "ht_away": ht_away, "ft_home": ft_home, "ft_away": ft_away, "match_id": match_id}
                    )
                else:
                    session.execute(
                        text("""
                            INSERT INTO scores (match_id, ht_home, ht_away, ft_home, ft_away, created_at)
                            VALUES (:match_id, :ht_home, :ht_away, :ft_home, :ft_away, NOW())
                        """),
                        {"match_id": match_id, "ht_home": ht_home, "ht_away": ht_away, "ft_home": ft_home, "ft_away": ft_away}
                    )
        
        session.commit()
        return is_new
    except Exception as e:
        session.rollback()
        raise e


def ingest_league(client: FootballDataClient, league_code: str, season: int = None) -> dict:
    """Ingest all matches for a league, return stats."""
    league_info = AVAILABLE_LEAGUES.get(league_code)
    if not league_info:
        print(f"  ✗ Unknown league code: {league_code}")
        return {"league": league_code, "error": f"Unknown league: {league_code}"}
    
    print(f"\n{'='*60}")
    print(f"Ingesting: {league_info['name']} ({league_code})")
    print(f"{'='*60}")
    
    stats = {
        "league": league_code,
        "name": league_info['name'],
        "matches_total": 0,
        "matches_new": 0,
        "matches_with_scores": 0,
    }
    
    # Team cache to avoid repeated lookups
    team_cache = {}
    
    try:
        # Open a fresh session for this league
        with get_session() as session:
            # Ensure league exists
            league_id = ensure_league_exists(
                session, 
                league_code, 
                league_info['name'], 
                league_info['country']
            )
            
            # Fetch matches for current season and optionally previous
            seasons_to_fetch = [season] if season else [2024, 2025]
            
            for s in seasons_to_fetch:
                print(f"\n  Fetching season {s}...")
                
                try:
                    matches = client.get_matches(league_code=league_code, season=s)
                    print(f"  Retrieved {len(matches)} matches")
                    
                    for match in matches:
                        try:
                            # Ensure teams exist
                            home_team_id = ensure_team_exists(session, match.home_team, league_id, team_cache)
                            away_team_id = ensure_team_exists(session, match.away_team, league_id, team_cache)
                            
                            # Ingest match
                            is_new = ingest_match(session, match, league_id, home_team_id, away_team_id)
                            
                            stats["matches_total"] += 1
                            if is_new:
                                stats["matches_new"] += 1
                            
                            # Count matches with half-time scores
                            if (match.status and match.status.value == 'FINISHED' and 
                                match.score and match.score.half_time and 
                                match.score.half_time.home is not None):
                                stats["matches_with_scores"] += 1
                        except Exception as e:
                            print(f"    ⚠ Error on match {match.id}: {e}")
                            continue
                    
                except Exception as e:
                    print(f"  ⚠ Error fetching season {s}: {e}")
                    continue
        
        print(f"\n  ✓ Completed: {stats['matches_total']} matches ({stats['matches_new']} new)")
        print(f"    Matches with HT scores: {stats['matches_with_scores']}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        stats["error"] = str(e)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Ingest football league data")
    parser.add_argument(
        "--leagues", 
        nargs="+", 
        default=DEFAULT_LEAGUES,
        choices=list(AVAILABLE_LEAGUES.keys()),
        help=f"Leagues to ingest. Available: {', '.join(AVAILABLE_LEAGUES.keys())}"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Specific season to ingest (e.g., 2024). Default: 2024 and 2025"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all available leagues"
    )
    
    args = parser.parse_args()
    
    leagues_to_ingest = list(AVAILABLE_LEAGUES.keys()) if args.all else args.leagues
    
    print("="*60)
    print("Multi-League Data Ingestion")
    print("="*60)
    print(f"Leagues to ingest: {', '.join(leagues_to_ingest)}")
    print(f"Season: {args.season or 'All (2024-2025)'}")
    print()
    
    # Initialize
    client = FootballDataClient()
    
    all_stats = []
    
    for league_code in leagues_to_ingest:
        stats = ingest_league(client, league_code, args.season)
        all_stats.append(stats)
    
    # Summary
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    
    total_matches = sum(s.get("matches_total", 0) for s in all_stats)
    total_new = sum(s.get("matches_new", 0) for s in all_stats)
    total_with_scores = sum(s.get("matches_with_scores", 0) for s in all_stats)
    
    print(f"\n{'League':<20} {'Total':<10} {'New':<10} {'With HT':<10}")
    print("-"*50)
    for s in all_stats:
        if "error" not in s:
            print(f"{s['league']:<20} {s['matches_total']:<10} {s['matches_new']:<10} {s['matches_with_scores']:<10}")
        else:
            print(f"{s['league']:<20} ERROR: {s.get('error', 'Unknown')}")
    
    print("-"*50)
    print(f"{'TOTAL':<20} {total_matches:<10} {total_new:<10} {total_with_scores:<10}")
    
    # Coverage stats
    if total_matches > 0:
        coverage = 100 * total_with_scores / total_matches
        print(f"\nHalf-time score coverage: {coverage:.1f}%")
    
    print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()
