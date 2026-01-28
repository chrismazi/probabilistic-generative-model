"""Check database tables."""
from src.db import get_session
from sqlalchemy import text

with get_session() as session:
    result = session.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
    tables = [r[0] for r in result.fetchall()]
    print(f"Tables in public schema: {tables}")
    
    if not tables:
        print("\nNo tables found. Creating schema...")
        
        # Create tables one by one
        sqls = [
            """CREATE TABLE IF NOT EXISTS leagues (
                id SERIAL PRIMARY KEY,
                code VARCHAR(10) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                country VARCHAR(100),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS teams (
                id SERIAL PRIMARY KEY,
                external_id INTEGER UNIQUE,
                name VARCHAR(100) NOT NULL,
                short_name VARCHAR(50),
                tla VARCHAR(5),
                crest_url TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS matches (
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
            )""",
            """CREATE TABLE IF NOT EXISTS scores (
                id SERIAL PRIMARY KEY,
                match_id INTEGER UNIQUE REFERENCES matches(id) ON DELETE CASCADE,
                ht_home INTEGER,
                ht_away INTEGER,
                ft_home INTEGER,
                ft_away INTEGER,
                winner VARCHAR(10),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )""",
        ]
        
        for sql in sqls:
            try:
                session.execute(text(sql))
                session.commit()
                print(f"  Created table")
            except Exception as e:
                print(f"  Error: {e}")
        
        # Check again
        result = session.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
        tables = [r[0] for r in result.fetchall()]
        print(f"\nTables after creation: {tables}")
