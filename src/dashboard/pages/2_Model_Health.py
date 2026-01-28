"""
Model Health Page - Diagnostics and calibration.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Model Health | Half-Goals",
    page_icon="🔬",
    layout="wide",
)

API_BASE_URL = "http://localhost:8000"


def get_api_data(endpoint: str, params: dict = None) -> dict:
    """Fetch data from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("⚽ Half-Goals Predictions")
st.sidebar.page_link("app.py", label="🎯 Upcoming Matches")
st.sidebar.page_link("pages/1_Recent_Results.py", label="📊 Recent Results")
st.sidebar.page_link("pages/2_Model_Health.py", label="🔬 Model Health")

st.sidebar.markdown("---")

# League selector
leagues_data = get_api_data("/api/v1/leagues")
if leagues_data:
    league_options = {f"{l['name']} ({l['code']})": l['code'] for l in leagues_data}
    selected_league_display = st.sidebar.selectbox(
        "Select League",
        options=list(league_options.keys()),
        index=0 if league_options else None,
    )
    selected_league = league_options.get(selected_league_display, "PL")
else:
    selected_league = "PL"


# =============================================================================
# Main Content
# =============================================================================

st.title("🔬 Model Health & Diagnostics")

# Fetch evaluation data
eval_data = get_api_data("/api/v1/evaluation/summary", params={"league": selected_league})
predictions_data = get_api_data("/api/v1/predictions/upcoming", params={"league": selected_league, "days_ahead": 7})

# =============================================================================
# Model Diagnostics Section
# =============================================================================

st.header("MCMC Diagnostics")

if predictions_data:
    diagnostics = predictions_data.get("model_diagnostics", {})
    is_healthy = diagnostics.get("is_healthy", False)
    fit_mode = diagnostics.get("fit_mode", "unknown")
    
    # Health status banner
    if is_healthy:
        st.success("✅ **Model is HEALTHY** - Predictions are reliable.")
    else:
        st.error(f"""
        ❌ **Model is UNHEALTHY** - Predictions may be unreliable.
        
        **Fit Mode:** {fit_mode.upper()}
        
        **Warning:** {diagnostics.get('warning', 'Unknown issue')}
        """)
    
    # Diagnostics metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rhat = diagnostics.get("max_rhat", 0)
        rhat_ok = rhat <= 1.01
        st.metric(
            "Max R-hat",
            f"{rhat:.3f}",
            delta="✓ OK" if rhat_ok else "✗ > 1.01",
            delta_color="normal" if rhat_ok else "inverse",
        )
    
    with col2:
        ess = diagnostics.get("min_ess", 0)
        ess_ok = ess >= 400
        st.metric(
            "Min ESS",
            f"{ess}",
            delta="✓ OK" if ess_ok else "✗ < 400",
            delta_color="normal" if ess_ok else "inverse",
        )
    
    with col3:
        divergences = diagnostics.get("n_divergences", 0)
        div_ok = divergences == 0
        st.metric(
            "Divergences",
            f"{divergences}",
            delta="✓ None" if div_ok else "✗ Bad",
            delta_color="normal" if div_ok else "inverse",
        )
    
    with col4:
        st.metric(
            "Fit Mode",
            fit_mode.upper(),
            delta="Production" if fit_mode == "production" else "Smoke Test",
            delta_color="normal" if fit_mode == "production" else "off",
        )
    
    st.markdown("""
    | Metric | Threshold | Description |
    |--------|-----------|-------------|
    | R-hat | ≤ 1.01 | Convergence diagnostic. Values > 1.01 indicate chains haven't converged. |
    | ESS | ≥ 400 | Effective sample size. Low values mean high autocorrelation. |
    | Divergences | 0 | Sampling problems. Any divergence indicates model specification issues. |
    """)
    
    # Versioning
    st.markdown("---")
    st.subheader("Model Version Info")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.code(predictions_data.get("model_version", "N/A"), language=None)
        st.caption("Model Version")
    with col2:
        st.code(predictions_data.get("data_cutoff_utc", "N/A"), language=None)
        st.caption("Data Cutoff (Training)")
    with col3:
        st.code(predictions_data.get("generated_at_utc", "N/A"), language=None)
        st.caption("Last Generated")

else:
    st.warning("Unable to fetch model diagnostics.")


# =============================================================================
# Evaluation Metrics Section
# =============================================================================

st.markdown("---")
st.header("📊 Evaluation Metrics")

if eval_data:
    metrics = eval_data.get("metrics", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        brier = metrics.get("brier_score", 0)
        st.metric(
            "Brier Score",
            f"{brier:.4f}",
            help="Lower is better. 0 = perfect, 0.25 = random."
        )
    
    with col2:
        log_loss = metrics.get("log_loss", 0)
        st.metric(
            "Log Loss",
            f"{log_loss:.4f}",
            help="Lower is better. Penalizes confident wrong predictions."
        )
    
    with col3:
        accuracy = metrics.get("accuracy", 0)
        st.metric(
            "Accuracy",
            f"{accuracy:.1%}",
            help="Percentage of correct predictions."
        )
    
    with col4:
        ece = metrics.get("calibration_ece", 0)
        st.metric(
            "ECE",
            f"{ece:.4f}",
            help="Expected Calibration Error. Lower is better."
        )
    
    # Baseline comparison
    st.markdown("---")
    st.subheader("Comparison vs Baselines")
    
    baselines = eval_data.get("vs_baselines", {})
    
    for baseline, result in baselines.items():
        if "better" in result.lower():
            st.success(f"✓ vs **{baseline}**: {result}")
        elif "worse" in result.lower():
            st.error(f"✗ vs **{baseline}**: {result}")
        else:
            st.info(f"● vs **{baseline}**: {result}")
    
    # Interpretation
    st.markdown("---")
    st.info(f"**Interpretation:** {eval_data.get('interpretation', 'N/A')}")
    
    # Recommendation
    model_status = eval_data.get("model_status", {})
    if not model_status.get("is_healthy", True):
        st.warning(f"**Recommendation:** {model_status.get('recommendation', 'Refit model.')}")

else:
    st.warning("Unable to fetch evaluation metrics.")


# =============================================================================
# Calibration Chart (Simulated)
# =============================================================================

st.markdown("---")
st.header("📈 Calibration Curve")

st.caption("Shows how well predicted probabilities match actual outcomes.")

import plotly.graph_objects as go

# Simulated calibration data (would come from API in production)
bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
predicted = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
observed = [0.08, 0.18, 0.28, 0.42, 0.48, 0.58, 0.72, 0.78, 0.88]  # Slightly off

fig = go.Figure()

# Perfect calibration line
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode="lines",
    line=dict(dash="dash", color="gray"),
    name="Perfect Calibration",
))

# Actual calibration
fig.add_trace(go.Scatter(
    x=predicted,
    y=observed,
    mode="lines+markers",
    marker=dict(size=10),
    line=dict(color="#3498db"),
    name="Model Calibration",
))

fig.update_layout(
    xaxis=dict(title="Predicted Probability", range=[0, 1], tickformat=".0%"),
    yaxis=dict(title="Observed Frequency", range=[0, 1], tickformat=".0%"),
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)

st.plotly_chart(fig, use_container_width=True)

st.caption("*Simulated calibration curve. In production, this would be computed from held-out validation data.*")


# =============================================================================
# Recommendations
# =============================================================================

st.markdown("---")
st.header("🛠️ Recommendations")

if predictions_data and not predictions_data.get("model_diagnostics", {}).get("is_healthy", True):
    st.markdown("""
    ### To improve model health:
    
    1. **Increase sampling:**
       ```python
       config = ModelConfig(
           n_samples=1000,
           n_tune=1000,
           n_chains=4,
       )
       ```
    
    2. **Run production fit:**
       ```bash
       python scripts/fit_model_jax.py
       ```
    
    3. **Verify diagnostics:**
       - R-hat ≤ 1.01
       - ESS ≥ 400
       - Divergences = 0
    """)
else:
    st.success("Model is healthy! No action needed.")


st.markdown("---")
st.caption("⚽ Football Half-Goals Prediction System | Dashboard v1.0")
