"""
Probity Debate Package

This package provides infrastructure for running debates with real-time deception probe scoring.
Supports both transparent debates (where debaters see probe scores) and oversight debates 
(where only judges see probe scores).
"""

from .types import (
    DebateType,
    ProviderType, 
    ModelConfig,
    ProbeConfig,
    DebateConfig,
    ConversationTurn,
    ProbeScore,
    DebateResult
)

from .config import (
    ProbeInferenceConfig,
    ConversationConfig,
    DebateManagerConfig,
    DefaultConfigs,
    DEBATE_TOPICS
)

__version__ = "0.1.0"
__author__ = "Soothcheck"

__all__ = [
    # Types
    "DebateType",
    "ProviderType",
    "ModelConfig", 
    "ProbeConfig",
    "DebateConfig",
    "ConversationTurn",
    "ProbeScore",
    "DebateResult",
    
    # Config
    "ProbeInferenceConfig",
    "ConversationConfig", 
    "DebateManagerConfig",
    "DefaultConfigs",
    "DEBATE_TOPICS"
]