"""
Phase 6.6 Step 3: Model Fitting on Real Data

Fits a Poisson model on Premier League historical data.
Uses single chain and proper multiprocessing guard for Windows.
"""

if __name__ == '__main__':
    from datetime import datetime
    from src.db import get_session
    from sqlalchemy import text
    import numpy as np

    from src.bayesian import HalfGoalModel, MatchData, TrainingData, ModelConfig

    print("=" * 60)
    print("PHASE 6.6 Step 3: Model Fitting")
    print("=" * 60)

    # Load training data from DB
    print("\n1. Loading training data...")

    with get_session() as session:
        # Get league ID
        result = session.execute(text("SELECT id FROM leagues WHERE code = 'PL'")).fetchone()
        league_id = result[0]
        
        # Get finished matches with scores
        result = session.execute(text("""
            SELECT 
                m.id,
                m.home_team_id,
                m.away_team_id,
                s.ht_home,
                s.ht_away,
                s.ft_home,
                s.ft_away
            FROM matches m
            JOIN scores s ON m.id = s.match_id
            WHERE m.league_id = :league_id
              AND m.status = 'FINISHED'
              AND s.ht_home IS NOT NULL
              AND s.ft_home IS NOT NULL
            ORDER BY m.kickoff_utc DESC
            LIMIT 400
        """), {"league_id": league_id}).fetchall()

    print(f"   Loaded {len(result)} matches")

    # Convert to MatchData
    matches = []
    for row in result:
        match_id, home_id, away_id, ht_h, ht_a, ft_h, ft_a = row
        matches.append(MatchData(
            match_id=match_id,
            league_idx=league_id,
            home_team_idx=home_id,
            away_team_idx=away_id,
            g1_home=ht_h,
            g1_away=ht_a,
            g2_home=ft_h - ht_h,
            g2_away=ft_a - ht_a,
        ))

    # Prepare training data
    data = TrainingData.from_matches(matches)
    print(f"   Training on {data.n_matches} matches")
    print(f"   Unique teams: {len(set(data.home_team_idx) | set(data.away_team_idx))}")

    # Quick data summary
    g1_total = data.g1_home + data.g1_away
    g2_total = data.g2_home + data.g2_away
    p_2h_gt_1h = np.mean(g2_total > g1_total)
    print(f"   Historical P(G2 > G1): {p_2h_gt_1h:.2%}")

    # Configure model - SINGLE CHAIN for Windows
    print("\n2. Fitting model (single chain for Windows)...")

    config = ModelConfig(
        model_type="poisson",
        n_samples=500,
        n_tune=300,
        n_chains=1,  # Single chain to avoid multiprocessing issues
        target_accept=0.9,
    )

    model = HalfGoalModel(config)

    import time
    start = time.time()
    model.fit(data)
    elapsed = time.time() - start
    print(f"   Fitting completed in {elapsed:.1f}s")

    # Get diagnostics
    print("\n3. Diagnostics...")
    diagnostics = model.get_diagnostics()
    print(f"   N Divergences: {diagnostics['n_divergences']}")
    print(f"   Max R-hat: {diagnostics['max_rhat']:.3f}")
    print(f"   Min ESS (bulk): {diagnostics['min_ess_bulk']:.0f}")
    print(f"   Is Healthy: {diagnostics['is_healthy']}")

    # Get posterior summary
    print("\n4. Posterior Summary...")
    summary = model.get_posterior_summary()
    ha = summary["home_advantage"]["multiplicative"]
    print(f"   Home Advantage: {ha['interpretation']}")
    print(f"     - Median: {ha['median']:.2f}x")
    print(f"     - 90% CI: [{ha['ci_5']:.2f}, {ha['ci_95']:.2f}]")

    alpha1 = summary["alpha_1"]
    alpha2 = summary["alpha_2"]
    print(f"   Expected goals per team per half:")
    print(f"     - 1H: {alpha1['expected_goals_per_team']:.2f}")
    print(f"     - 2H: {alpha2['expected_goals_per_team']:.2f}")

    # Save model
    print("\n5. Saving model...")
    from pathlib import Path
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / f"model_poisson_{datetime.now().strftime('%Y%m%d_%H%M')}.nc"
    model.save(str(model_path))
    print(f"   Saved to {model_path}")

    print("\n✓ Model fitting complete!")
    print("=" * 60)
