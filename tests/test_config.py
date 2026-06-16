import os
import pytest
from unittest.mock import patch

def test_config_loads_from_env():
    with patch.dict(os.environ, {
        "TOGETHER_API_KEY": "test-key",
        "HF_TOKEN": "hf-test",
        "DATABASE_URL": "postgresql://localhost/test",
    }):
        # reimport to pick up patched env
        import importlib
        import app.config as cfg
        importlib.reload(cfg)
        settings = cfg.Settings()
        assert settings.together_api_key == "test-key"
        assert settings.hf_token == "hf-test"
        assert settings.database_url == "postgresql://localhost/test"
