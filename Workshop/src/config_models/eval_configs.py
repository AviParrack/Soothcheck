# Configuration models for model evaluation/benchmarking system
# Defines structures for specifying evaluation datasets, model selections,
# and response formats

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import hashlib


class EvalPrompt(BaseModel):
    """Single evaluation prompt"""

    prompt: str
    prompt_id: Optional[str] = Field(
        None, description="Unique identifier for the prompt"
    )
    # Additional fields can be added later (e.g., expected response, tags, etc.)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def get_id(self) -> str:
        """Get prompt ID, generating one if not set"""
        if self.prompt_id:
            return self.prompt_id
        # Generate a stable ID from prompt content
        return hashlib.md5(self.prompt.encode()).hexdigest()[:12]


class EvalDataset(BaseModel):
    """Configuration for an evaluation dataset"""

    name: str = Field(..., description="Name of the evaluation dataset")
    prompts: List[EvalPrompt] = Field(..., description="List of evaluation prompts")
    description: Optional[str] = Field(None, description="Description of the dataset")

    @classmethod
    def from_simple_json(cls, data: Dict[str, Any], name: str) -> "EvalDataset":
        """Create from simple JSON format with just a list of prompt objects"""
        prompts = []
        prompt_list = data if isinstance(data, list) else data.get("prompts", [])

        for idx, item in enumerate(prompt_list):
            if isinstance(item, str):
                prompts.append(EvalPrompt(prompt=item))
            elif isinstance(item, dict) and "prompt" in item:
                # Extract prompt and any additional metadata
                prompt_text = item.pop("prompt")
                prompt_id = item.pop("prompt_id", None)
                prompts.append(
                    EvalPrompt(prompt=prompt_text, prompt_id=prompt_id, metadata=item)
                )
            else:
                raise ValueError(f"Invalid prompt format: {item}")

        return cls(
            name=name,
            prompts=prompts,
            description=data.get("description") if isinstance(data, dict) else None,
        )


class ModelSpec(BaseModel):
    """Specification for a model to evaluate"""

    source_type: str = Field(
        ...,
        description="Type of source: 'suite_experiment', 'run_path', or 'base_model'",
    )
    source_path: str = Field(
        ...,
        description="Path to the model (suite/experiment_id, dataset/run_name, or model name)",
    )
    display_name: Optional[str] = Field(
        None, description="Optional display name for the model"
    )


class EvalConfig(BaseModel):
    """Configuration for an evaluation run"""

    # Model selection
    experiment_suites: List[str] = Field(
        default_factory=list, description="List of experiment suite names to evaluate"
    )
    additional_models: List[ModelSpec] = Field(
        default_factory=list, description="Additional models to evaluate beyond suites"
    )

    # Dataset
    eval_dataset_path: str = Field(
        ..., description="Path to evaluation dataset JSON file"
    )

    # Output
    output_dir: str = Field(..., description="Directory to save evaluation results")

    # Execution settings
    batch_size: int = Field(default=1, description="Batch size for inference")
    max_new_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Nucleus sampling threshold")

    def get_all_model_specs(self) -> List[ModelSpec]:
        """Get all model specifications from suites and additional models"""
        specs = list(self.additional_models)

        # Add suite experiments (will be expanded during execution)
        for suite_name in self.experiment_suites:
            specs.append(
                ModelSpec(
                    source_type="suite",
                    source_path=suite_name,
                    display_name=f"Suite: {suite_name}",
                )
            )

        return specs


class ModelResponsePair(BaseModel):
    """Single prompt-response pair"""

    prompt: str
    response: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ModelResponses(BaseModel):
    """Complete set of responses from a model"""

    # Model identification
    model_id: str = Field(..., description="Unique identifier for this model run")
    model_path: str = Field(..., description="Full path to the model")
    model_type: str = Field(..., description="Type of model: trained, base, etc.")
    run_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Training run metadata if available"
    )

    # Evaluation metadata
    eval_dataset_name: str = Field(..., description="Name of evaluation dataset")
    eval_timestamp: datetime = Field(default_factory=datetime.now)
    eval_config: Dict[str, Any] = Field(
        ..., description="Evaluation configuration used"
    )

    # Model configuration details
    base_model: str = Field(..., description="Base model name")
    peft_applied: bool = Field(
        default=False, description="Whether PEFT adapters were applied"
    )
    peft_config: Optional[Dict[str, Any]] = Field(
        None, description="PEFT configuration if applicable"
    )

    # Responses
    responses: List[ModelResponsePair] = Field(
        ..., description="List of prompt-response pairs"
    )

    # Performance metrics
    total_prompts: int = Field(..., description="Total number of prompts evaluated")
    successful_responses: int = Field(..., description="Number of successful responses")
    average_response_length: Optional[float] = Field(
        None, description="Average response length in tokens"
    )
    total_time_seconds: Optional[float] = Field(
        None, description="Total evaluation time"
    )


class ComparisonPair(BaseModel):
    """A pair of model responses to compare"""

    prompt_id: str
    prompt: str
    model_a_id: str
    model_b_id: str
    response_a: str
    response_b: str


class JudgmentResult(BaseModel):
    """Result of judging a comparison pair"""

    comparison_id: str = Field(..., description="Unique ID for this comparison")
    prompt_id: str
    model_a_id: str
    model_b_id: str
    winner: Literal["A", "B", "tie"]
    judge_type: str = Field(
        ..., description="Type of judge (e.g., 'gemini-2.0-pro', 'human')"
    )
    judge_explanation: Optional[str] = Field(
        None, description="Explanation for the judgment"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class JudgingSession(BaseModel):
    """Tracks progress of a judging session"""

    dataset_name: str
    session_id: str
    judge_type: str
    total_comparisons: int
    completed_comparisons: int
    judgments: List[JudgmentResult] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    def get_remaining_comparisons(
        self, all_pairs: List[ComparisonPair]
    ) -> List[ComparisonPair]:
        """Get comparisons that haven't been judged yet"""
        completed_ids = {j.comparison_id for j in self.judgments}
        return [
            pair
            for pair in all_pairs
            if self._get_comparison_id(pair) not in completed_ids
        ]

    @staticmethod
    def _get_comparison_id(pair: ComparisonPair) -> str:
        """Generate stable ID for a comparison"""
        # Sort model IDs to ensure same pair always gets same ID
        models = sorted([pair.model_a_id, pair.model_b_id])
        return f"{pair.prompt_id}_{models[0]}_{models[1]}"


class ModelWinStats(BaseModel):
    """Win/loss statistics for a model"""

    model_id: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_comparisons: int = 0
    elo_rating: Optional[float] = Field(
        None, description="ELO rating calculated from win/loss"
    )

    @property
    def win_rate(self) -> float:
        if self.total_comparisons == 0:
            return 0.0
        return self.wins / self.total_comparisons


class ComparisonResults(BaseModel):
    """Complete results of model comparisons for a dataset"""

    dataset_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    total_models: int
    total_comparisons: int
    judge_types: List[str]
    model_stats: Dict[str, ModelWinStats]
    all_judgments: List[JudgmentResult]
    elo_ratings: Optional[Dict[str, float]] = Field(
        None, description="Final ELO ratings"
    )
