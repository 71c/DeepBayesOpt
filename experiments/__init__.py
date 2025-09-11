"""
Experiment Management Module

Centralized experiment registry and execution system for Bayesian optimization research.
"""

from .registry import ExperimentRegistry, get_registry
from .runner import ExperimentRunner

__all__ = ['ExperimentRegistry', 'get_registry', 'ExperimentRunner']