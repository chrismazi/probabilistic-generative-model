# Probabilistic Generative Model for football match Tips

A probabilistic model for predicting P(G₂ > G₁) — the probability that the second half of a football match has more goals than the first half.

## Features

- **Hierarchical Bayesian modeling** with partial pooling across leagues
- **Proper uncertainty quantification** via posterior sampling
- **Time-inhomogeneous Poisson process** for goal timing
- **Strict evaluation** with time-split backtesting and calibration checks
- **Decision layer** that only acts when there's measurable edge

## Tech Stack

- **Database**: PostgreSQL
- **Inference**: PyMC
- **API**: FastAPI
- **Dashboard**: Streamlit
- **Data Source**: football-data.org

## Quick Start

```bash
# Clone and install
git clone <repo>
cd probabilistic-generative-model
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your settings

# Initialize database
python scripts/init_db.py

# Run daily ingestion
python scripts/daily_ingest.py

# Start API
uvicorn src.api.main:app --reload

# Start dashboard
streamlit run src/dashboard/app.py
```

## Project Structure

```
src/
├── config.py           # Configuration management
├── data/               # API client, ingestion, quality checks
├── features/           # Rolling windows, Elo, feature builders
├── models/             # Bayesian models, priors, prediction
├── evaluation/         # Metrics, backtesting, calibration
├── decision/           # Thresholds, Kelly staking, audit
├── api/                # FastAPI endpoints
└── dashboard/          # Streamlit visualization
```

## Model Overview

### v1: Joint Team-Level Hierarchical Model

```
G1_home ~ Poisson(μ1_h)   G1_away ~ Poisson(μ1_a)
G2_home ~ Poisson(μ2_h)   G2_away ~ Poisson(μ2_a)

log(μ) = H_L + a_i - d_j + league_effect
```

With sum-to-zero constraints for identifiability.

### v2: Splines + Dixon-Coles (after v1 calibrated)

- League-specific minute-level intensity splines
- Low-score correlation adjustment

## License

MIT
