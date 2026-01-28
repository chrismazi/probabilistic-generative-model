"""
Streamlit Dashboard for Football Half-Goals Predictions.

This dashboard ONLY reads from the API. It does NOT:
- Fit models
- Ingest data
- Run MCMC
- Touch raw DB tables directly

All data comes from: http://localhost:8080/api/v1/...
"""

import streamlit as st

# Page config must be first
st.set_page_config(
    page_title="Football Half-Goals Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

import requests
from datetime import datetime
import pandas as pd

# =============================================================================
# API Configuration
# =============================================================================

API_BASE_URL = "http://localhost:8080"


def get_api_data(endpoint: str, params: dict = None) -> dict:
    """Fetch data from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Start it with: `uvicorn src.api.main:app`")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("⚽ Half-Goals Predictions")
st.sidebar.markdown("---")

# API status
api_health = get_api_data("/api/v1/health")
if api_health:
    status = api_health.get("status", "unknown")
    if status == "ok":
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.warning(f"⚠️ API Status: {status}")
else:
    st.sidebar.error("❌ API Offline")

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

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard reads from API only. No model fitting.")


# =============================================================================
# Main Page - Upcoming Matches
# =============================================================================

st.title("🎯 Upcoming Match Predictions")

# Fetch predictions
predictions_data = get_api_data(
    "/api/v1/predictions/upcoming",
    params={"league": selected_league, "days_ahead": 7}
)

if predictions_data:
    # Model health warning
    diagnostics = predictions_data.get("model_diagnostics", {})
    is_healthy = diagnostics.get("is_healthy", False)
    fit_mode = diagnostics.get("fit_mode", "unknown")
    
    # Warning banner
    if not is_healthy:
        st.warning(f"""
        ⚠️ **Model Health Warning** - {fit_mode.upper()} FIT MODE
        
        {diagnostics.get('warning', 'Model diagnostics below production thresholds.')}
        
        **Do not trust predictions for real decisions until model is healthy.**
        """)
    else:
        st.success("✅ Model is healthy and ready for production use.")
    
    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", predictions_data.get("total_matches", 0))
    with col2:
        st.metric("Signals", predictions_data.get("signals_count", 0))
    with col3:
        st.metric("Bets", predictions_data.get("bets_count", 0))
    with col4:
        st.metric("Fit Mode", fit_mode.upper())
    
    st.markdown("---")
    
    # Predictions table
    predictions = predictions_data.get("predictions", [])
    
    if predictions:
        st.subheader("📊 Predictions")
        
        # Build dataframe
        rows = []
        for p in predictions:
            uncertainty = p.get("p_2h_gt_1h", {})
            rows.append({
                "Match ID": p["match_id"],
                "Home": p["home_team"],
                "Away": p["away_team"],
                "Kickoff (UTC)": p["kickoff_utc"][:16].replace("T", " "),
                "P(2H > 1H)": f"{uncertainty.get('mean', 0):.1%}",
                "90% CI": f"[{uncertainty.get('ci_5', 0):.1%}, {uncertainty.get('ci_95', 0):.1%}]",
                "Decision": p["decision"].upper(),
                "Reason": p["decision_reason"][:50] + "..." if len(p["decision_reason"]) > 50 else p["decision_reason"],
            })
        
        df = pd.DataFrame(rows)
        
        # Style the decision column
        def style_decision(val):
            if val == "SIGNAL":
                return "background-color: #2ecc71; color: white"
            elif val == "BET":
                return "background-color: #3498db; color: white"
            elif val == "SKIP":
                return "background-color: #95a5a6; color: white"
            return ""
        
        st.dataframe(
            df.style.applymap(style_decision, subset=["Decision"]),
            use_container_width=True,
            hide_index=True,
        )
        
        # Detailed view
        st.markdown("---")
        st.subheader("🔍 Match Details")
        
        match_options = {f"{p['home_team']} vs {p['away_team']}": p for p in predictions}
        selected_match_name = st.selectbox("Select a match for details", options=list(match_options.keys()))
        
        if selected_match_name:
            match = match_options[selected_match_name]
            uncertainty = match.get("p_2h_gt_1h", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {match['home_team']} vs {match['away_team']}")
                st.markdown(f"**Kickoff:** {match['kickoff_utc'][:16].replace('T', ' ')} UTC")
                st.markdown(f"**Match ID:** {match['match_id']}")
                
                # Decision badge
                decision = match["decision"].upper()
                if decision == "SIGNAL":
                    st.success(f"🟢 Decision: **{decision}**")
                elif decision == "BET":
                    st.info(f"🔵 Decision: **{decision}**")
                else:
                    st.warning(f"⚪ Decision: **{decision}**")
                
                st.markdown(f"**Reason:** {match['decision_reason']}")
            
            with col2:
                st.markdown("### Probability Distribution")
                
                mean = uncertainty.get("mean", 0.5)
                ci_5 = uncertainty.get("ci_5", 0.4)
                ci_95 = uncertainty.get("ci_95", 0.6)
                std = uncertainty.get("std", 0.1)
                
                # Simple bar chart for probability
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # CI bar
                fig.add_trace(go.Bar(
                    x=[ci_95 - ci_5],
                    y=["P(2H > 1H)"],
                    orientation="h",
                    base=[ci_5],
                    marker_color="rgba(52, 152, 219, 0.3)",
                    name="90% CI",
                    hoverinfo="skip",
                ))
                
                # Mean marker
                fig.add_trace(go.Scatter(
                    x=[mean],
                    y=["P(2H > 1H)"],
                    mode="markers",
                    marker=dict(size=20, color="#e74c3c", symbol="diamond"),
                    name=f"Mean: {mean:.1%}",
                ))
                
                # 50% reference line
                fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
                
                fig.update_layout(
                    xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
                    yaxis=dict(showticklabels=False),
                    height=150,
                    margin=dict(l=20, r=20, t=20, b=40),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                | Statistic | Value |
                |-----------|-------|
                | Mean | {mean:.1%} |
                | Std Dev | {std:.3f} |
                | 5th percentile | {ci_5:.1%} |
                | 95th percentile | {ci_95:.1%} |
                | CI Width | {(ci_95 - ci_5):.1%} |
                """)
    else:
        st.info(predictions_data.get("message", "No upcoming matches found."))
    
    # Versioning info
    st.markdown("---")
    st.caption(f"""
    **Generated:** {predictions_data.get('generated_at_utc', 'N/A')} | 
    **Model:** {predictions_data.get('model_version', 'N/A')} | 
    **Data Cutoff:** {predictions_data.get('data_cutoff_utc', 'N/A')}
    """)

else:
    st.error("Unable to fetch predictions. Check if API is running.")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.caption("⚽ Football Half-Goals Prediction System | Dashboard v1.0")
