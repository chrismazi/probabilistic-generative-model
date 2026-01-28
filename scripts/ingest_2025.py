"""
Ingest 2025 season for upcoming matches.
"""

from datetime import date, datetime
from src.config import settings
from src.data.client import FootballDataClient
from src.db import get_session
from sqlalchemy import text

print("Ingesting 2025 season...")

client = FootballDataClient(settings.football_data_api_key)
api_matches = client.get_matches("PL", season=2025)
print(f"Fetched {len(api_matches)} matches")

def get_ht(m):
    if m.score and m.score.half_time:
        return m.score.half_time.home, m.score.half_time.away
    return None, None

def get_ft(m):
    if m.score and m.score.full_time:
        return m.score.full_time.home, m.score.full_time.away
    return None, None

with get_session() as session:
    # Get league ID
    result = session.execute(text("SELECT id FROM leagues WHERE code = 'PL'")).fetchone()
    league_id = result[0]
    
    # Get team map
    result = session.execute(text("SELECT external_id, id FROM teams"))
    team_id_map = {r[0]: r[1] for r in result.fetchall()}
    
    matches_inserted = 0
    scores_inserted = 0
    teams_inserted = 0
    
    for match in api_matches:
        # Ensure teams exist
        for team in [match.home_team, match.away_team]:
            if team.id not in team_id_map:
                session.execute(
                    text("INSERT INTO teams (external_id, name, short_name, tla) VALUES (:ext_id, :name, :short_name, :tla)"),
                    {"ext_id": team.id, "name": team.name, "short_name": team.short_name, "tla": team.tla}
                )
                session.commit()
                result = session.execute(text("SELECT id FROM teams WHERE external_id = :ext_id"), {"ext_id": team.id}).fetchone()
                team_id_map[team.id] = result[0]
                teams_inserted += 1
        
        # Check if match exists
        result = session.execute(text("SELECT id FROM matches WHERE external_id = :ext_id"), {"ext_id": match.id}).fetchone()
        
        if result:
            match_db_id = result[0]
        else:
            season_str = str(match.season.get("startDate", "2025")[:4]) if match.season else "2025"
            
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
            result = session.execute(text("SELECT id FROM matches WHERE external_id = :ext_id"), {"ext_id": match.id}).fetchone()
            match_db_id = result[0]
            matches_inserted += 1
        
        # Insert scores if finished
        ft_h, ft_a = get_ft(match)
        if match.status.value == "FINISHED" and ft_h is not None:
            result = session.execute(text("SELECT id FROM scores WHERE match_id = :match_id"), {"match_id": match_db_id}).fetchone()
            
            if not result:
                ht_h, ht_a = get_ht(match)
                winner = match.score.winner if match.score else None
                
                session.execute(
                    text("""
                        INSERT INTO scores (match_id, ht_home, ht_away, ft_home, ft_away, winner)
                        VALUES (:match_id, :ht_home, :ht_away, :ft_home, :ft_away, :winner)
                    """),
                    {"match_id": match_db_id, "ht_home": ht_h, "ht_away": ht_a, "ft_home": ft_h, "ft_away": ft_a, "winner": winner}
                )
                session.commit()
                scores_inserted += 1

print(f"Inserted: {teams_inserted} teams, {matches_inserted} matches, {scores_inserted} scores")

# Verify
with get_session() as session:
    result = session.execute(text("SELECT COUNT(*) FROM matches WHERE league_id = :lid"), {"lid": league_id}).fetchone()
    print(f"\nTotal matches in DB: {result[0]}")
    
    result = session.execute(text("""
        SELECT COUNT(*) FROM matches m 
        WHERE m.league_id = :lid AND m.status IN ('SCHEDULED', 'TIMED')
          AND m.kickoff_utc > NOW()
    """), {"lid": league_id}).fetchone()
    print(f"Upcoming matches: {result[0]}")
    
    result = session.execute(text("""
        SELECT COUNT(*) FROM matches m 
        JOIN scores s ON m.id = s.match_id 
        WHERE m.league_id = :lid AND s.ht_home IS NOT NULL
    """), {"lid": league_id}).fetchone()
    print(f"Matches with HT scores: {result[0]}")
