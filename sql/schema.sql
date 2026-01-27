-- =============================================================================
-- Probabilistic Generative Model - Database Schema
-- PostgreSQL DDL for betting predictions system
-- =============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- LEAGUES
-- =============================================================================
CREATE TABLE leagues (
    id SERIAL PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL,           -- e.g., 'PL', 'BL1', 'SA'
    name VARCHAR(100) NOT NULL,                 -- e.g., 'Premier League'
    country VARCHAR(50) NOT NULL,               -- e.g., 'England'
    provider_key VARCHAR(50),                   -- football-data.org key
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_leagues_code ON leagues(code);
CREATE INDEX idx_leagues_active ON leagues(is_active) WHERE is_active = TRUE;

-- =============================================================================
-- TEAMS
-- =============================================================================
CREATE TABLE teams (
    id SERIAL PRIMARY KEY,
    league_id INTEGER REFERENCES leagues(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    short_name VARCHAR(20),
    provider_team_id INTEGER,                   -- football-data.org team ID
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(league_id, provider_team_id)
);

CREATE INDEX idx_teams_league ON teams(league_id);
CREATE INDEX idx_teams_provider_id ON teams(provider_team_id);

-- =============================================================================
-- MATCHES
-- =============================================================================
CREATE TYPE match_status AS ENUM (
    'SCHEDULED',
    'TIMED', 
    'IN_PLAY',
    'PAUSED',
    'FINISHED',
    'POSTPONED',
    'SUSPENDED',
    'CANCELLED',
    'AWARDED'
);

CREATE TABLE matches (
    id SERIAL PRIMARY KEY,
    league_id INTEGER REFERENCES leagues(id) ON DELETE CASCADE,
    season VARCHAR(10) NOT NULL,                -- e.g., '2024-25'
    matchday INTEGER,
    kickoff_utc TIMESTAMPTZ NOT NULL,           -- ALL TIMES IN UTC
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    status match_status DEFAULT 'SCHEDULED',
    provider_match_id INTEGER UNIQUE,           -- football-data.org match ID
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_matches_league ON matches(league_id);
CREATE INDEX idx_matches_kickoff ON matches(kickoff_utc);
CREATE INDEX idx_matches_season ON matches(league_id, season);
CREATE INDEX idx_matches_status ON matches(status);
CREATE INDEX idx_matches_teams ON matches(home_team_id, away_team_id);

-- =============================================================================
-- SCORES
-- =============================================================================
CREATE TABLE scores (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE UNIQUE,
    ht_home INTEGER,                            -- Half-time home goals
    ht_away INTEGER,                            -- Half-time away goals
    ft_home INTEGER,                            -- Full-time home goals
    ft_away INTEGER,                            -- Full-time away goals
    ht_available BOOLEAN DEFAULT FALSE,         -- Flag for coverage tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints: HT <= FT
    CONSTRAINT chk_ht_home CHECK (ht_home IS NULL OR ft_home IS NULL OR ht_home <= ft_home),
    CONSTRAINT chk_ht_away CHECK (ht_away IS NULL OR ft_away IS NULL OR ht_away <= ft_away),
    CONSTRAINT chk_non_negative CHECK (
        (ht_home IS NULL OR ht_home >= 0) AND
        (ht_away IS NULL OR ht_away >= 0) AND
        (ft_home IS NULL OR ft_home >= 0) AND
        (ft_away IS NULL OR ft_away >= 0)
    )
);

CREATE INDEX idx_scores_match ON scores(match_id);

-- =============================================================================
-- MATCH FEATURES (computed features per match)
-- =============================================================================
CREATE TABLE match_features (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    feature_version VARCHAR(20) NOT NULL,       -- e.g., 'v1.0'
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Team rolling features (JSONB for flexibility)
    home_features JSONB NOT NULL,
    away_features JSONB NOT NULL,
    
    -- League-level features
    league_features JSONB NOT NULL,
    
    -- Metadata
    home_match_count INTEGER,                   -- How many matches used
    away_match_count INTEGER,
    is_valid BOOLEAN DEFAULT TRUE,              -- Met minimum history requirements
    
    UNIQUE(match_id, feature_version)
);

CREATE INDEX idx_match_features_match ON match_features(match_id);
CREATE INDEX idx_match_features_valid ON match_features(is_valid) WHERE is_valid = TRUE;

-- =============================================================================
-- MODEL VERSIONS (track model iterations)
-- =============================================================================
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) UNIQUE NOT NULL,        -- e.g., 'v1.0.0'
    model_type VARCHAR(50) NOT NULL,            -- e.g., 'hierarchical_poisson'
    description TEXT,
    
    -- Model configuration
    config JSONB NOT NULL,                      -- Hyperparameters, priors
    
    -- Reproducibility
    random_seed INTEGER,
    git_sha VARCHAR(40),
    data_snapshot_id VARCHAR(100),
    
    -- Fit summary
    fit_summary JSONB,                          -- Convergence diagnostics, WAIC, etc.
    
    -- Timestamps
    trained_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE             -- Currently deployed
);

CREATE INDEX idx_model_versions_active ON model_versions(is_active) WHERE is_active = TRUE;

-- =============================================================================
-- PREDICTIONS
-- =============================================================================
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    model_version_id INTEGER REFERENCES model_versions(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Core prediction: P(G2 > G1)
    p_2h_gt_1h FLOAT NOT NULL,
    p_2h_gt_1h_ci_low FLOAT,                    -- 5th percentile
    p_2h_gt_1h_ci_high FLOAT,                   -- 95th percentile
    
    -- Other probabilities
    p_1h_gt_2h FLOAT,
    p_equal FLOAT,
    
    -- Uncertainty metrics
    entropy FLOAT,                              -- Prediction uncertainty
    
    -- Explanation (top drivers)
    explanation JSONB,
    
    UNIQUE(match_id, model_version_id)
);

CREATE INDEX idx_predictions_match ON predictions(match_id);
CREATE INDEX idx_predictions_model ON predictions(model_version_id);
CREATE INDEX idx_predictions_prob ON predictions(p_2h_gt_1h);

-- =============================================================================
-- DECISIONS (audit log for decision layer)
-- =============================================================================
CREATE TYPE decision_rule AS ENUM (
    'edge_threshold',
    'confidence_threshold',
    'uncertainty_filter',
    'minimum_history',
    'no_action'
);

CREATE TYPE decision_outcome AS ENUM (
    'WIN',
    'LOSE',
    'VOID',
    'PENDING'
);

CREATE TABLE decisions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    prediction_id INTEGER REFERENCES predictions(id),
    model_version_id INTEGER REFERENCES model_versions(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Decision inputs
    p FLOAT NOT NULL,                           -- Model probability
    p_ci_low FLOAT,
    p_ci_high FLOAT,
    p_be FLOAT,                                 -- Break-even probability (1/odds)
    odds FLOAT,                                 -- Decimal odds if available
    
    -- Decision output
    rule_triggered decision_rule NOT NULL,
    action_taken BOOLEAN DEFAULT FALSE,         -- Did we actually recommend?
    stake FLOAT,                                -- NULL in no-stake mode
    
    -- Outcome (filled post-match)
    outcome decision_outcome DEFAULT 'PENDING',
    pnl FLOAT,                                  -- Profit/loss if staked
    
    UNIQUE(match_id, model_version_id)
);

CREATE INDEX idx_decisions_match ON decisions(match_id);
CREATE INDEX idx_decisions_outcome ON decisions(outcome);
CREATE INDEX idx_decisions_action ON decisions(action_taken) WHERE action_taken = TRUE;

-- =============================================================================
-- COVERAGE REPORT (track data quality per league)
-- =============================================================================
CREATE TABLE coverage_reports (
    id SERIAL PRIMARY KEY,
    league_id INTEGER REFERENCES leagues(id) ON DELETE CASCADE,
    season VARCHAR(10) NOT NULL,
    report_date DATE NOT NULL,
    
    -- Coverage metrics
    total_matches INTEGER,
    matches_with_ht INTEGER,
    ht_coverage_pct FLOAT,
    avg_delay_hours FLOAT,                      -- Avg delay until HT available
    
    -- Quality flags
    is_reliable BOOLEAN,                        -- HT coverage >= 90%
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(league_id, season, report_date)
);

CREATE INDEX idx_coverage_league ON coverage_reports(league_id, season);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to compute G2 > G1 from scores
CREATE OR REPLACE FUNCTION compute_2h_gt_1h(
    ht_home INTEGER,
    ht_away INTEGER,
    ft_home INTEGER,
    ft_away INTEGER
) RETURNS BOOLEAN AS $$
DECLARE
    g1 INTEGER;
    g2 INTEGER;
BEGIN
    IF ht_home IS NULL OR ht_away IS NULL OR ft_home IS NULL OR ft_away IS NULL THEN
        RETURN NULL;
    END IF;
    
    g1 := ht_home + ht_away;
    g2 := (ft_home - ht_home) + (ft_away - ht_away);
    
    RETURN g2 > g1;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- =============================================================================
-- VIEWS
-- =============================================================================

-- Complete match view with computed outcomes
CREATE VIEW match_outcomes AS
SELECT 
    m.id AS match_id,
    l.code AS league_code,
    m.season,
    m.kickoff_utc,
    ht.name AS home_team,
    at.name AS away_team,
    s.ht_home,
    s.ht_away,
    s.ft_home,
    s.ft_away,
    (s.ht_home + s.ht_away) AS g1_total,
    ((s.ft_home - s.ht_home) + (s.ft_away - s.ht_away)) AS g2_total,
    compute_2h_gt_1h(s.ht_home, s.ht_away, s.ft_home, s.ft_away) AS actual_2h_gt_1h,
    m.status
FROM matches m
JOIN leagues l ON m.league_id = l.id
JOIN teams ht ON m.home_team_id = ht.id
JOIN teams at ON m.away_team_id = at.id
LEFT JOIN scores s ON m.id = s.match_id;

-- League baseline P(G2 > G1)
CREATE VIEW league_baselines AS
SELECT 
    l.code AS league_code,
    m.season,
    COUNT(*) AS total_matches,
    SUM(CASE WHEN compute_2h_gt_1h(s.ht_home, s.ht_away, s.ft_home, s.ft_away) THEN 1 ELSE 0 END) AS g2_gt_g1_count,
    AVG(CASE WHEN compute_2h_gt_1h(s.ht_home, s.ht_away, s.ft_home, s.ft_away) THEN 1.0 ELSE 0.0 END) AS p_2h_gt_1h
FROM matches m
JOIN leagues l ON m.league_id = l.id
JOIN scores s ON m.id = s.match_id
WHERE m.status = 'FINISHED'
  AND s.ht_home IS NOT NULL
GROUP BY l.code, m.season;
