"""
PhysGym: Environment for Physics Discovery

This package provides environments for physics equation discovery
tasks, where agents learn to design experiments and propose hypotheses to uncover
underlying physical laws.
"""
__version__ = "0.1.0"

# Core classes
from .phyenv import PhyEnv
from .interface import ResearchInterface, ExperimentRunState, setup_logging

# Utility functions
from .utils.metrics import evaluate_hypothesis
from .utils.sandbox import create_function_from_string
from .utils.llm_providers import (
    get_recommended_provider,
    show_provider_status,
    load_api_key,
)

__all__ = [
    # Core classes
    "PhyEnv",
    "ResearchInterface",
    "ExperimentRunState",
    "setup_logging",
    # Utility functions
    "evaluate_hypothesis",
    "create_function_from_string",
    "get_recommended_provider",
    "show_provider_status",
    "load_api_key",
]