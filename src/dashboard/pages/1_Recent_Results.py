"""
Recent Results Page - Shows completed matches with outcomes.
"""

import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Recent Results | Half-Goals",
    page_icon="⚽",
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

limit = st.sidebar.slider("Number of results", 10, 100, 30)


# =============================================================================
# Main Content
# =============================================================================

st.title("📊 Recent Match Results")
st.markdown("Completed matches with actual outcomes for G1 vs G2 comparison.")

results_data = get_api_data(
    "/api/v1/results/recent",
    params={"league": selected_league, "limit": limit}
)

if results_data:
    # Summary stats
    outcomes = [r["outcome"] for r in results_data]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(results_data))
    with col2:
        count_2h = outcomes.count("2H_MORE")
        st.metric("2H > 1H", f"{count_2h} ({100*count_2h/len(outcomes):.1f}%)")
    with col3:
        count_1h = outcomes.count("1H_MORE")
        st.metric("1H > 2H", f"{count_1h} ({100*count_1h/len(outcomes):.1f}%)")
    with col4:
        count_eq = outcomes.count("EQUAL")
        st.metric("Equal", f"{count_eq} ({100*count_eq/len(outcomes):.1f}%)")
    
    st.markdown("---")
    
    # Results table
    rows = []
    for r in results_data:
        rows.append({
            "Date": r["kickoff_utc"][:10],
            "Home": r["home_team"],
            "Away": r["away_team"],
            "HT": f"{r['ht_home']}-{r['ht_away']}",
            "FT": f"{r['ft_home']}-{r['ft_away']}",
            "G1 (1H)": r["g1_total"],
            "G2 (2H)": r["g2_total"],
            "Outcome": r["outcome"],
        })
    
    df = pd.DataFrame(rows)
    
    # Style outcomes
    def style_outcome(val):
        if val == "2H_MORE":
            return "background-color: #2ecc71; color: white"
        elif val == "1H_MORE":
            return "background-color: #e74c3c; color: white"
        else:
            return "background-color: #f1c40f; color: black"
    
    st.dataframe(
        df.style.applymap(style_outcome, subset=["Outcome"]),
        use_container_width=True,
        hide_index=True,
    )
    
    # Distribution chart
    st.markdown("---")
    st.subheader("Outcome Distribution")
    
    import plotly.express as px
    
    outcome_counts = pd.DataFrame({
        "Outcome": ["2H > 1H", "1H > 2H", "Equal"],
        "Count": [count_2h, count_1h, count_eq],
        "Color": ["#2ecc71", "#e74c3c", "#f1c40f"],
    })
    
    fig = px.bar(
        outcome_counts,
        x="Outcome",
        y="Count",
        color="Outcome",
        color_discrete_map={"2H > 1H": "#2ecc71", "1H > 2H": "#e74c3c", "Equal": "#f1c40f"},
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No recent results available.")

st.markdown("---")
st.caption("⚽ Football Half-Goals Prediction System | Dashboard v1.0")
