"""
Simple ingestion test for Phase 6.6.

Ingests one league (PL) and reports coverage.
"""

from datetime import date, datetime
from src.config import settings
from src.data.client import FootballDataClient
from src.db import get_session
from sqlalchemy import text
import json

print("=" * 60)
print("PHASE 6.6: Data Ingestion Test")
print("=" * 60)

# Create client
client = FootballDataClient(settings.football_data_api_key)

# Get PL matches for current season
print("\n1. Fetching Premier League matches (2024 season)...")
api_matches = client.get_matches("PL", season=2024)
print(f"   Fetched {len(api_matches)} matches")

# Count finished vs scheduled
finished = [m for m in api_matches if m.status.value == "FINISHED"]
scheduled = [m for m in api_matches if m.status.value in ["SCHEDULED", "TIMED"]]
print(f"   - Finished: {len(finished)}")
print(f"   - Scheduled: {len(scheduled)}")

# Helper functions to access ApiMatch data correctly
def get_ht(m):
    """Get half-time scores."""
    if m.score and m.score.half_time:
        return m.score.half_time.home, m.score.half_time.away
    return None, None

def get_ft(m):
    """Get full-time scores."""
    if m.score and m.score.full_time:
        return m.score.full_time.home, m.score.full_time.away
    return None, None

def has_ht(m):
    """Check if HT scores available."""
    ht_h, ht_a = get_ht(m)
    return ht_h is not None and ht_a is not None

# Check HT scores coverage
with_ht = [m for m in finished if has_ht(m)]
print(f"   - Finished with HT scores: {len(with_ht)} ({100*len(with_ht)/max(len(finished),1):.1f}%)")

# Store in database
print("\n2. Storing in database...")

with get_session() as session:
    # First, ensure league exists
    result = session.execute(
        text("SELECT id FROM leagues WHERE code = 'PL'")
    ).fetchone()
    
    if result:
        league_id = result[0]
        print(f"   League PL already exists (id={league_id})")
    else:
        session.execute(
            text("INSERT INTO leagues (code, name, country) VALUES ('PL', 'Premier League', 'England') RETURNING id")
        )
        session.commit()
        result = session.execute(text("SELECT id FROM leagues WHERE code = 'PL'")).fetchone()
        league_id = result[0]
        print(f"   Created league PL (id={league_id})")
    
    # Insert teams
    teams_inserted = 0
    team_id_map = {}
    
    for match in api_matches:
        for team in [match.home_team, match.away_team]:
            ext_id = team.id
            if ext_id in team_id_map:
                continue
            
            # Check if exists
            result = session.execute(
                text("SELECT id FROM teams WHERE external_id = :ext_id"),
                {"ext_id": ext_id}
            ).fetchone()
            
            if result:
                team_id_map[ext_id] = result[0]
            else:
                session.execute(
                    text("INSERT INTO teams (external_id, name, short_name, tla) VALUES (:ext_id, :name, :short_name, :tla)"),
                    {"ext_id": ext_id, "name": team.name, "short_name": team.short_name, "tla": team.tla}
                )
                session.commit()
                result = session.execute(
                    text("SELECT id FROM teams WHERE external_id = :ext_id"),
                    {"ext_id": ext_id}
                ).fetchone()
                team_id_map[ext_id] = result[0]
                teams_inserted += 1
    
    print(f"   Inserted {teams_inserted} new teams")
    
    # Insert matches
    matches_inserted = 0
    scores_inserted = 0
    
    for match in api_matches:
        # Check if exists
        result = session.execute(
            text("SELECT id FROM matches WHERE external_id = :ext_id"),
            {"ext_id": match.id}
        ).fetchone()
        
        if result:
            match_db_id = result[0]
        else:
            # Get season from season dict
            season_str = str(match.season.get("startDate", "2024")[:4]) if match.season else "2024"
            
            session.execute(
                text("""
                    INSERT INTO matches (external_id, league_id, season, matchday, 
                        home_team_id, away_team_id, kickoff_utc, status)
                    VALUES (:ext_id, :league_id, :season, :matchday, 
                        :home_id, :away_id, :kickoff, :status)
                """),
                {
                    "ext_id": match.id,
                    "league_id": league_id,
                    "season": season_str,
                    "matchday": match.matchday,
                    "home_id": team_id_map[match.home_team.id],
                    "away_id": team_id_map[match.away_team.id],
                    "kickoff": match.utc_date,
                    "status": match.status.value,
                }
            )
            session.commit()
            result = session.execute(
                text("SELECT id FROM matches WHERE external_id = :ext_id"),
                {"ext_id": match.id}
            ).fetchone()
            match_db_id = result[0]
            matches_inserted += 1
        
        # Insert scores if finished
        ft_h, ft_a = get_ft(match)
        if match.status.value == "FINISHED" and ft_h is not None:
            # Check if exists
            result = session.execute(
                text("SELECT id FROM scores WHERE match_id = :match_id"),
                {"match_id": match_db_id}
            ).fetchone()
            
            if not result:
                ht_h, ht_a = get_ht(match)
                winner = match.score.winner if match.score else None
                
                session.execute(
                    text("""
                        INSERT INTO scores (match_id, ht_home, ht_away, ft_home, ft_away, winner)
                        VALUES (:match_id, :ht_home, :ht_away, :ft_home, :ft_away, :winner)
                    """),
                    {
                        "match_id": match_db_id,
                        "ht_home": ht_h,
                        "ht_away": ht_a,
                        "ft_home": ft_h,
                        "ft_away": ft_a,
                        "winner": winner,
                    }
                )
                session.commit()
                scores_inserted += 1
    
    print(f"   Inserted {matches_inserted} new matches")
    print(f"   Inserted {scores_inserted} new scores")

# Verify
print("\n3. Verification...")
with get_session() as session:
    result = session.execute(text("SELECT COUNT(*) FROM matches WHERE league_id = :lid"), {"lid": league_id}).fetchone()
    print(f"   Total matches in DB: {result[0]}")
    
    result = session.execute(text("""
        SELECT COUNT(*) FROM matches m 
        JOIN scores s ON m.id = s.match_id 
        WHERE m.league_id = :lid AND s.ht_home IS NOT NULL
    """), {"lid": league_id}).fetchone()
    print(f"   Matches with HT scores: {result[0]}")
    
    result = session.execute(text("""
        SELECT COUNT(*) FROM matches m 
        WHERE m.league_id = :lid AND m.status = 'FINISHED'
    """), {"lid": league_id}).fetchone()
    print(f"   Finished matches: {result[0]}")
    
    result = session.execute(text("""
        SELECT COUNT(*) FROM matches m 
        WHERE m.league_id = :lid AND m.status IN ('SCHEDULED', 'TIMED')
          AND m.kickoff_utc > NOW()
    """), {"lid": league_id}).fetchone()
    print(f"   Upcoming matches: {result[0]}")

print("\n✓ Ingestion complete!")
print("=" * 60)
