from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Literal, Any
from enum import Enum


class DebateType(Enum):
    """Types of debates supported"""
    TRANSPARENT_DEBATE = "transparent_debate"  # Both debaters see probe scores
    TRANSPARENT_OVERSIGHT = "transparent_oversight"  # Only judge sees probe scores
    ASYMMETRIC_TRANSPARENT = "asymmetric_transparent"  # Only one debater sees scores


class ProviderType(Enum):
    """Supported model providers"""
    LOCAL = "local"  # HuggingFace/TransformerLens
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    REPLICATE = "replicate"


@dataclass
class ModelConfig:
    """Configuration for a model"""
    provider: ProviderType
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    generation_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.generation_kwargs is None:
            self.generation_kwargs = {}


@dataclass
class ProbeConfig:
    """Configuration for probe scoring"""
    probe_dir: str
    probe_types: List[str]
    layer: int
    model_name: str  # Model being probed (usually local)
    device: str = "cuda"
    enabled: bool = True


@dataclass
class DebateConfig:
    """Configuration for a debate"""
    debater1: ModelConfig
    debater2: ModelConfig
    judge: ModelConfig
    probe_config: Optional[ProbeConfig]
    debate_type: DebateType
    max_rounds: int = 6
    topic: Optional[str] = None
    save_dir: str = "./debate_results"


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""
    speaker: Literal["debater1", "debater2", "judge", "system"]
    content: str
    timestamp: float
    probe_scores: Optional[Dict[str, List[float]]] = None  # probe_type -> token_scores
    tokens: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProbeScore:
    """Probe scoring results for a response"""
    probe_type: str
    layer: int
    tokens: List[str]
    token_scores: List[float]
    mean_score: float
    metadata: Dict[str, Any]


@dataclass
class DebateResult:
    """Complete debate result"""
    debate_id: str
    config: DebateConfig
    conversation: List[ConversationTurn]
    winner: Optional[Literal["debater1", "debater2", "tie"]]
    judge_reasoning: Optional[str]
    probe_analysis: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    start_time: float
    end_time: float