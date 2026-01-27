"""
Tests for configuration module.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import DecisionMode, Settings, get_settings


class TestSettings:
    """Test configuration loading and validation."""
    
    def test_default_settings(self):
        """Test default settings are valid."""
        settings = Settings()
        
        assert settings.database_pool_size == 5
        assert settings.min_matches_required == 5
        assert settings.random_seed == 42
        assert settings.decision_mode == DecisionMode.PROBABILITY_ONLY
    
    def test_decision_mode_enum(self):
        """Test decision mode enum values."""
        assert DecisionMode.PROBABILITY_ONLY.value == "probability_only"
        assert DecisionMode.STAKE_ENABLED.value == "stake_enabled"
    
    def test_edge_threshold_validation(self):
        """Test edge threshold bounds."""
        # Valid
        settings = Settings(edge_threshold=0.1)
        assert settings.edge_threshold == 0.1
        
        # At bounds
        settings = Settings(edge_threshold=0.0)
        assert settings.edge_threshold == 0.0
        
        settings = Settings(edge_threshold=0.5)
        assert settings.edge_threshold == 0.5
    
    def test_confidence_threshold_validation(self):
        """Test confidence threshold bounds."""
        settings = Settings(confidence_threshold=0.9)
        assert settings.confidence_threshold == 0.9
    
    def test_kelly_fraction_validation(self):
        """Test Kelly fraction bounds."""
        settings = Settings(kelly_fraction=0.25)
        assert settings.kelly_fraction == 0.25
    
    def test_cache_dir_creation(self, tmp_path):
        """Test cache directory is created."""
        cache_dir = tmp_path / "test_cache"
        settings = Settings(cache_dir=cache_dir)
        
        assert settings.cache_dir == cache_dir
        assert cache_dir.exists()
    
    def test_log_file_parent_creation(self, tmp_path):
        """Test log file parent directory is created."""
        log_file = tmp_path / "logs" / "app.log"
        settings = Settings(log_file=log_file)
        
        assert settings.log_file == log_file
        assert log_file.parent.exists()
    
    @patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test_key_123"})
    def test_env_loading(self):
        """Test settings load from environment."""
        # Clear cache to pick up new env
        get_settings.cache_clear()
        settings = get_settings()
        
        assert settings.football_data_api_key == "test_key_123"
        
        # Restore cache
        get_settings.cache_clear()


class TestDecisionMode:
    """Test decision mode validation."""
    
    def test_probability_only_is_default(self):
        """Verify probability_only is the safe default."""
        settings = Settings()
        assert settings.decision_mode == DecisionMode.PROBABILITY_ONLY
    
    def test_stake_enabled_requires_explicit_set(self):
        """Stake mode must be explicitly enabled."""
        settings = Settings(decision_mode=DecisionMode.STAKE_ENABLED)
        assert settings.decision_mode == DecisionMode.STAKE_ENABLED
