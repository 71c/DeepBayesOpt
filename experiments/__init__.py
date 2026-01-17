"""
Experiment Management Module

Centralized experiment registry and execution system for Bayesian optimization research.
"""

from .registry import ExperimentRegistry, REGISTRY
from .runner import ExperimentRunner

__all__ = ['ExperimentRegistry', 'REGISTRY', 'ExperimentRunner']