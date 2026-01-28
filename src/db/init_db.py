"""
Database initialization script.

Creates all tables needed for the betting model.
"""

from src.db import get_session
from sqlalchemy import text


SCHEMA_SQL = """
-- Leagues
CREATE TABLE IF NOT EXISTS leagues (
    id SERIAL PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    country VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Teams
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    external_id INTEGER UNIQUE,
    name VARCHAR(100) NOT NULL,
    short_name VARCHAR(50),
    tla VARCHAR(5),
    crest_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Matches
CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    external_id INTEGER UNIQUE,
    league_id INTEGER REFERENCES leagues(id),
    season VARCHAR(10),
    matchday INTEGER,
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    kickoff_utc TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'SCHEDULED',
    venue TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Scores
CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    match_id INTEGER UNIQUE REFERENCES matches(id) ON DELETE CASCADE,
    ht_home INTEGER,
    ht_away INTEGER,
    ft_home INTEGER,
    ft_away INTEGER,
    winner VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Team Features (rolling window stats)
CREATE TABLE IF NOT EXISTS team_features (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id),
    league_id INTEGER REFERENCES leagues(id),
    as_of_date DATE NOT NULL,
    window_size INTEGER DEFAULT 10,
    matches_in_window INTEGER DEFAULT 0,
    
    -- Goals
    goals_scored_avg FLOAT DEFAULT 0,
    goals_conceded_avg FLOAT DEFAULT 0,
    goals_scored_1h_avg FLOAT DEFAULT 0,
    goals_scored_2h_avg FLOAT DEFAULT 0,
    goals_conceded_1h_avg FLOAT DEFAULT 0,
    goals_conceded_2h_avg FLOAT DEFAULT 0,
    
    -- Half patterns
    rate_2h_gt_1h FLOAT DEFAULT 0,
    
    -- Attack/Defense
    attack_strength FLOAT DEFAULT 1.0,
    defense_strength FLOAT DEFAULT 1.0,
    
    -- Timestamps
    feature_version INTEGER DEFAULT 1,
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(team_id, as_of_date, window_size)
);

-- Elo Ratings
CREATE TABLE IF NOT EXISTS elo_ratings (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id),
    league_id INTEGER REFERENCES leagues(id),
    rating FLOAT DEFAULT 1500,
    matches_played INTEGER DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(team_id, league_id)
);

-- Model Versions
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) UNIQUE NOT NULL,
    model_type VARCHAR(50),
    config JSONB,
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE
);

-- Predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    model_version_id INTEGER REFERENCES model_versions(id),
    
    p_2h_gt_1h FLOAT,
    p_2h_gt_1h_ci_low FLOAT,
    p_2h_gt_1h_ci_high FLOAT,
    entropy FLOAT,
    explanation JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(match_id, model_version_id)
);

-- Decisions (audit log)
CREATE TABLE IF NOT EXISTS decisions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    decision_type VARCHAR(30) NOT NULL,
    decision_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    p_predicted FLOAT,
    p_ci_low FLOAT,
    p_ci_high FLOAT,
    
    odds FLOAT,
    stake_fraction FLOAT,
    expected_value FLOAT,
    prob_exceeds_breakeven FLOAT,
    
    actual_outcome INTEGER,
    realized_pnl FLOAT,
    
    model_version VARCHAR(20),
    explanation TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_id);
CREATE INDEX IF NOT EXISTS idx_matches_kickoff ON matches(kickoff_utc);
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);
CREATE INDEX IF NOT EXISTS idx_scores_match ON scores(match_id);
CREATE INDEX IF NOT EXISTS idx_team_features_team ON team_features(team_id, as_of_date);
CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_decisions_match ON decisions(match_id);
"""


def init_database():
    """Create all database tables."""
    print("Initializing database schema...")
    
    # Split by statements
    statements = [s.strip() for s in SCHEMA_SQL.split(';') if s.strip() and not s.strip().startswith('--')]
    
    for statement in statements:
        try:
            with get_session() as session:
                session.execute(text(statement))
                session.commit()
        except Exception as e:
            # Only warn for index errors, fail for table errors
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                pass  # Ignore
            elif "CREATE INDEX" in statement:
                print(f"Warning (index): {e}")
            else:
                print(f"Error: {e}")
                raise
    
    print("Database schema created successfully!")
    
    # Verify tables
    with get_session() as session:
        result = session.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
        ))
        tables = [r[0] for r in result.fetchall()]
        print(f"Tables created: {tables}")


if __name__ == "__main__":
    init_database()
