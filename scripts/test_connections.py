"""Quick test script for DB and API connectivity."""
from src.db import get_session
from sqlalchemy import text
from src.config import settings

print("Testing database connection...")
try:
    with get_session() as session:
        result = session.execute(text('SELECT 1'))
        print('  DB connection: OK')
        
        # Check tables
        result = session.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
        ))
        tables = [r[0] for r in result.fetchall()]
        print(f'  Tables found: {len(tables)}')
        print(f'  Tables: {tables}')
except Exception as e:
    print(f'  DB connection FAILED: {e}')

print("\nTesting API connection...")
if settings.football_data_api_key:
    try:
        from src.data.client import FootballDataClient
        client = FootballDataClient(settings.football_data_api_key)
        leagues = client.get_leagues()
        print(f'  API connection: OK')
        print(f'  Leagues available: {len(leagues)}')
        for l in leagues[:5]:
            print(f'    - {l.code}: {l.name}')
    except Exception as e:
        print(f'  API connection FAILED: {e}')
else:
    print('  No API key configured')
