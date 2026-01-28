"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root(self, client):
        """Root endpoint returns ok."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
    
    def test_health(self, client):
        """Health check returns status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data


class TestLeaguesEndpoint:
    """Test leagues endpoint."""
    
    def test_get_leagues(self, client):
        """Get leagues returns list."""
        response = client.get("/api/v1/leagues")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestPredictionsEndpoint:
    """Test predictions endpoint."""
    
    def test_predictions_upcoming(self, client):
        """Get upcoming predictions."""
        response = client.get("/api/v1/predictions/upcoming?league=PL")
        
        # Should always return 200 now (empty case handled gracefully)
        assert response.status_code == 200
        
        data = response.json()
        assert "model_diagnostics" in data
        assert "predictions" in data
        assert "generated_at_utc" in data
        assert "data_cutoff_utc" in data
        assert "message" in data
        
        # Check model diagnostics structure
        diagnostics = data["model_diagnostics"]
        assert "is_healthy" in diagnostics
        assert "fit_mode" in diagnostics  # New field
        assert "n_divergences" in diagnostics
        assert "max_rhat" in diagnostics
        assert "min_ess" in diagnostics
    
    def test_predictions_response_has_uncertainty(self, client):
        """Predictions include uncertainty."""
        response = client.get("/api/v1/predictions/upcoming?league=PL")
        
        if response.status_code == 200:
            data = response.json()
            if data["predictions"]:
                pred = data["predictions"][0]
                
                # Check uncertainty structure
                p = pred["p_2h_gt_1h"]
                assert "mean" in p
                assert "ci_5" in p
                assert "ci_95" in p
                assert "std" in p


class TestResultsEndpoint:
    """Test results endpoint."""
    
    def test_recent_results(self, client):
        """Get recent results."""
        response = client.get("/api/v1/results/recent?league=PL&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if data:
            result = data[0]
            assert "match_id" in result
            assert "home_team" in result
            assert "away_team" in result
            assert "g1_total" in result
            assert "g2_total" in result
            assert "outcome" in result


class TestEvaluationEndpoint:
    """Test evaluation endpoint."""
    
    def test_evaluation_summary(self, client):
        """Get evaluation summary."""
        response = client.get("/api/v1/evaluation/summary?league=PL")
        assert response.status_code == 200
        data = response.json()
        
        assert "metrics" in data
        assert "model_status" in data
