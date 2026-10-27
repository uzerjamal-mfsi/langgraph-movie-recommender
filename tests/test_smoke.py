import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from langgraph_movie_recommender.config import Settings
from langgraph_movie_recommender.graph.workflow import build_workflow

APP_MODULES = [
    "langgraph_movie_recommender",
    "langgraph_movie_recommender.cli",
    "langgraph_movie_recommender.config",
    "langgraph_movie_recommender.main",
    "langgraph_movie_recommender.models",
    "langgraph_movie_recommender.tmdb",
    "langgraph_movie_recommender.db",
    "langgraph_movie_recommender.db.models",
    "langgraph_movie_recommender.db.repository",
    "langgraph_movie_recommender.db.session",
    "langgraph_movie_recommender.graph.llm",
    "langgraph_movie_recommender.graph.nodes",
    "langgraph_movie_recommender.graph.nodes.classification",
    "langgraph_movie_recommender.graph.nodes.common",
    "langgraph_movie_recommender.graph.nodes.feedback",
    "langgraph_movie_recommender.graph.nodes.preferences",
    "langgraph_movie_recommender.graph.nodes.recommender",
    "langgraph_movie_recommender.graph.nodes.routing",
    "langgraph_movie_recommender.graph.nodes.taste",
    "langgraph_movie_recommender.graph.nodes.titles",
    "langgraph_movie_recommender.graph.prompts",
    "langgraph_movie_recommender.graph.recommendations",
    "langgraph_movie_recommender.graph.schemas",
    "langgraph_movie_recommender.graph.state",
    "langgraph_movie_recommender.graph.structured",
    "langgraph_movie_recommender.graph.workflow",
    "langgraph_movie_recommender.log_config",
    "langgraph_movie_recommender.memory.vector_store",
]

REQUIRED_ENV_VARS = {
    "OPENROUTER_API_KEY": "test-openrouter-key",
    "OPENROUTER_MODEL": "test/model",
    "TMDB_API_KEY": "test-tmdb-key",
}

OPTIONAL_ENV_VARS = (
    "TMDB_ACCESS_TOKEN",
    "DATA_DIR",
    "LIBRARY_DB_PATH",
    "CHECKPOINT_DB_PATH",
    "LOG_LEVEL",
)


def set_required_env(monkeypatch):
    for name, value in REQUIRED_ENV_VARS.items():
        monkeypatch.setenv(name, value)
    for name in OPTIONAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("module_name", APP_MODULES)
def test_application_modules_import(module_name):
    assert importlib.import_module(module_name) is not None


def test_workflow_compiles_without_external_calls():
    assert build_workflow() is not None


def test_settings_load_required_values(monkeypatch):
    set_required_env(monkeypatch)

    settings = Settings(_env_file=None)

    assert settings.openrouter_api_key == "test-openrouter-key"
    assert settings.openrouter_model == "test/model"
    assert settings.tmdb_api_key == "test-tmdb-key"
    assert settings.log_level == "INFO"


def test_settings_require_api_keys(monkeypatch):
    for name in (*REQUIRED_ENV_VARS, *OPTIONAL_ENV_VARS):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_default_data_paths(monkeypatch):
    set_required_env(monkeypatch)

    settings = Settings(_env_file=None)

    assert settings.data_dir == Path("data")
    assert settings.library_db_path == Path("data/library.db")
    assert settings.checkpoint_db_path == Path("data/checkpoints.db")
    assert settings.embedding_dimensions is None
