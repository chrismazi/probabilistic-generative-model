"""
Phase 6.6: Quick Evaluation (without full Bayesian model)

Uses historical base rates as simple predictions 
to verify evaluation pipeline works.
"""

if __name__ == '__main__':
    from datetime import datetime
    from src.db import get_session
    from sqlalchemy import text
    import numpy as np
    import json
    from pathlib import Path
    
    from src.evaluation import ModelEvaluator, BrierScore, CalibrationResult
    from src.decision import DecisionEngine, EXPERIMENTAL_CONFIG, DecisionOutcome
    
    print("=" * 60)
    print("PHASE 6.6: Quick Evaluation Pipeline Test")
    print("=" * 60)
    
    # Load historical data
    print("\n1. Loading historical data...")
    
    with get_session() as session:
        result = session.execute(text("SELECT id FROM leagues WHERE code = 'PL'")).fetchone()
        league_id = result[0]
        
        result = session.execute(text("""
            SELECT 
                m.id,
                s.ht_home,
                s.ht_away,
                s.ft_home,
                s.ft_away
            FROM matches m
            JOIN scores s ON m.id = s.match_id
            WHERE m.league_id = :league_id
              AND m.status = 'FINISHED'
              AND s.ht_home IS NOT NULL
            ORDER BY m.kickoff_utc
        """), {"league_id": league_id}).fetchall()
    
    print(f"   Loaded {len(result)} matches with scores")
    
    # Compute outcomes
    match_ids = np.array([r[0] for r in result])
    g1 = np.array([r[1] + r[2] for r in result])  # HT total
    g2 = np.array([(r[3] - r[1]) + (r[4] - r[2]) for r in result])  # 2H total
    outcomes = (g2 > g1).astype(int)
    
    print(f"   Historical P(G2 > G1): {outcomes.mean():.2%}")
    
    # Split into train/test (80/20)
    split_idx = int(len(outcomes) * 0.8)
    train_outcomes = outcomes[:split_idx]
    test_outcomes = outcomes[split_idx:]
    test_match_ids = match_ids[split_idx:]
    
    print(f"   Train: {split_idx} matches, Test: {len(test_outcomes)} matches")
    
    # "Model": Use training set base rate as prediction
    base_rate = train_outcomes.mean()
    predictions = np.full(len(test_outcomes), base_rate)
    print(f"   Simple model: predict P={base_rate:.3f} for all matches")
    
    # Evaluate
    print("\n2. Running evaluation...")
    
    evaluator = ModelEvaluator()
    result = evaluator.evaluate_with_baselines(
        predictions=predictions,
        outcomes=test_outcomes,
        model_name="base_rate_model",
    )
    
    print(f"\n   Model Comparison:")
    print(f"   {'Model':<20} {'Brier':>10} {'Log Loss':>10} {'Acc':>8}")
    print(f"   {'-'*50}")
    
    for name, eval_result in sorted(result.evaluations.items(), key=lambda x: x[1].brier_score.score):
        print(f"   {name:<20} {eval_result.brier_score.score:>10.4f} {eval_result.log_loss:>10.4f} {eval_result.accuracy:>8.1%}")
    
    print(f"\n   Best model: {result.best_model}")
    
    # Calibration
    print("\n3. Calibration check...")
    calibration = CalibrationResult.compute(predictions, test_outcomes)
    print(f"   ECE: {calibration.expected_calibration_error:.4f}")
    print(f"   MCE: {calibration.maximum_calibration_error:.4f}")
    
    # Decision layer
    print("\n4. Decision layer test...")
    
    engine = DecisionEngine(EXPERIMENTAL_CONFIG)
    decisions = []
    
    for i in range(min(20, len(test_match_ids))):
        match_id = int(test_match_ids[i])
        p_mean = base_rate + np.random.uniform(-0.1, 0.1)
        p_ci = (max(0.1, p_mean - 0.1), min(0.9, p_mean + 0.1))
        
        decision = engine.make_decision(
            match_id=match_id,
            p_mean=p_mean,
            p_ci=p_ci,
            odds=None,  # No odds
        )
        decisions.append(decision)
    
    signal_count = sum(1 for d in decisions if d.decision == DecisionOutcome.SIGNAL)
    skip_count = sum(1 for d in decisions if d.decision == DecisionOutcome.SKIP)
    print(f"   Made {len(decisions)} decisions:")
    print(f"     - SIGNAL: {signal_count}")
    print(f"     - SKIP: {skip_count}")
    
    # Save artifacts
    print("\n5. Saving artifacts...")
    
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Predictions artifact
    predictions_artifact = {
        "date": datetime.now().isoformat(),
        "model": "base_rate_model",
        "n_predictions": len(predictions),
        "predictions": [
            {"match_id": int(mid), "p_mean": float(p), "outcome": int(o)}
            for mid, p, o in zip(test_match_ids, predictions, test_outcomes)
        ][:50],  # First 50
    }
    
    pred_path = artifacts_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions_artifact, f, indent=2)
    print(f"   Saved {pred_path}")
    
    # Decisions artifact
    decisions_artifact = {
        "date": datetime.now().isoformat(),
        "config": "EXPERIMENTAL",
        "decisions": [d.to_dict() for d in decisions],
    }
    
    dec_path = artifacts_dir / f"decisions_{datetime.now().strftime('%Y%m%d')}.json"
    with open(dec_path, "w") as f:
        json.dump(decisions_artifact, f, indent=2)
    print(f"   Saved {dec_path}")
    
    # Evaluation report
    eval_report = {
        "date": datetime.now().isoformat(),
        "train_size": split_idx,
        "test_size": len(test_outcomes),
        "models": {},
    }
    
    for name, eval_result in result.evaluations.items():
        eval_report["models"][name] = {
            "brier_score": eval_result.brier_score.score,
            "log_loss": eval_result.log_loss,
            "accuracy": eval_result.accuracy,
            "ece": eval_result.calibration.expected_calibration_error,
        }
    
    eval_report["best_model"] = result.best_model
    
    eval_path = artifacts_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(eval_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"   Saved {eval_path}")
    
    print("\n" + "=" * 60)
    print("✓ PHASE 6.6 PIPELINE TEST COMPLETE")
    print("=" * 60)
    print("\nArtifacts generated:")
    print(f"  - {pred_path}")
    print(f"  - {dec_path}")
    print(f"  - {eval_path}")
