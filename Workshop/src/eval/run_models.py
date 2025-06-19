# Model loading and inference for evaluation/benchmarking
# Handles loading trained models with PEFT adapters and running inference

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm

from src.training.run_manager import RunManager, find_run
from src.utils.model_resolver import resolve_model_specification, ModelResolution
from src.config_models.eval_configs import (
    EvalDataset,
    ModelResponses,
    ModelResponsePair,
    ModelSpec,
)
from src.utils.litellm_logger import litellm_completion
from src.inference.inference import (
    preload_model_and_tokenizer,
    generate_batch_responses,
)


def load_model_from_spec(model_spec: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load a model using the project's standard model resolution and inference logic.

    This function uses the unified model loading from src.inference.inference
    to ensure consistency across the project. It provides extensive logging
    for clarity about which model is being loaded.

    Args:
        model_spec: Model specification string (config name, experiment path, or HF ID)

    Returns:
        Tuple of (model, tokenizer, metadata_dict)
    """
    print(f"\n{'='*80}")
    print(f"Loading model from specification: {model_spec}")
    print(f"{'='*80}")

    # Use the official model resolver to get model information
    try:
        resolution = resolve_model_specification(model_spec)
    except ValueError as e:
        # If resolution fails, it might be a direct HuggingFace ID
        print(
            f"Could not resolve as config/experiment, treating as HuggingFace ID: {model_spec}"
        )
        # For direct HF IDs, create minimal metadata
        model, tokenizer = preload_model_and_tokenizer(model_spec, device="auto")
        metadata_dict = {
            "model_spec": model_spec,
            "base_model": model_spec,
            "model_type": "chat",  # Default assumption for HF models
            "peft_applied": False,
            "peft_config": None,
            "is_experiment": False,
            "model_config": {
                "model_name_or_path": model_spec,
                "model_type": "chat",
            },
            "run_metadata": None,
        }
        return model, tokenizer, metadata_dict

    print(f"Resolution successful:")
    print(f"  - Is experiment: {resolution.is_experiment}")
    print(f"  - Base model: {resolution.base_model_hf_id}")
    print(f"  - Model type: {resolution.model_type}")

    # Extract metadata before loading
    run_metadata = None
    if resolution.is_experiment and resolution.experiment_path:
        print(f"  - Experiment path: {resolution.experiment_path}")
        metadata_path = resolution.experiment_path / "run_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                run_metadata = json.load(f)
            print(f"  - Found run metadata")

    # Check for PEFT config if this is an experiment
    peft_config_dict = None
    if resolution.peft_adapter_path:
        print(f"  - PEFT adapter path: {resolution.peft_adapter_path}")
        peft_config_path = Path(resolution.peft_adapter_path) / "adapter_config.json"
        if not peft_config_path.exists():
            peft_config_path = Path(resolution.peft_adapter_path) / "peft_config.json"

        if peft_config_path.exists():
            with open(peft_config_path, "r") as f:
                peft_config_dict = json.load(f)
            print(f"  - PEFT type: {peft_config_dict.get('peft_type', 'unknown')}")
            print(f"  - Target modules: {peft_config_dict.get('target_modules', [])}")

    # Load model using unified inference logic
    print(f"\n{'='*40}")
    print(f"LOADING MODEL WITH UNIFIED INFERENCE LOGIC")
    print(f"{'='*40}")

    model, tokenizer = preload_model_and_tokenizer(model_spec, device="auto")

    print(f"\n{'='*40}")
    print(f"MODEL LOADING COMPLETE")
    print(f"{'='*40}")

    # Prepare comprehensive metadata
    metadata_dict = {
        "model_spec": model_spec,
        "base_model": resolution.base_model_hf_id,
        "model_type": resolution.model_type,
        "peft_applied": bool(resolution.peft_adapter_path),
        "peft_config": peft_config_dict,
        "is_experiment": resolution.is_experiment,
        "experiment_path": (
            str(resolution.experiment_path) if resolution.experiment_path else None
        ),
        "model_config": resolution.model_config.model_dump(),
        "run_metadata": run_metadata,
    }

    print(f"\nModel metadata summary:")
    print(f"  - Base model: {metadata_dict['base_model']}")
    print(f"  - Model type: {metadata_dict['model_type']}")
    print(f"  - PEFT applied: {metadata_dict['peft_applied']}")
    print(f"  - From experiment: {metadata_dict['is_experiment']}")
    print(f"{'='*80}\n")

    return model, tokenizer, metadata_dict


def run_inference_on_prompts(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    model_name: str = "Model",
    model_type: str = "chat",
    show_progress: bool = True,
) -> List[str]:
    """
    Run inference on a list of prompts using the unified inference logic.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        model_name: Name for display/logging
        model_type: Type of model ("chat" or "text_generation")
        show_progress: Whether to show progress bar during inference

    Returns:
        List of generated responses
    """
    print(f"\n{'='*60}")
    print(f"RUNNING INFERENCE: {model_name}")
    print(f"{'='*60}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Model type: {model_type}")
    print(f"Generation settings:")
    print(f"  - max_new_tokens: {max_new_tokens}")
    print(f"  - temperature: {temperature}")
    print(f"  - top_p: {top_p}")
    print(f"{'='*60}\n")

    # Use the unified batch inference function
    responses = generate_batch_responses(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        prompt_name=None,  # No system prompt for evaluations
        show_progress=show_progress,
    )

    # Log summary
    successful_responses = sum(1 for r in responses if not r.startswith("ERROR:"))
    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE: {model_name}")
    print(f"Successful responses: {successful_responses}/{len(prompts)}")
    print(f"{'='*60}\n")

    return responses


def evaluate_model_on_dataset(
    model_spec: ModelSpec,
    eval_dataset: EvalDataset,
    eval_config: Dict[str, Any],
    output_dir: str,
) -> ModelResponses:
    """
    Evaluate a single model on the evaluation dataset.

    Args:
        model_spec: Specification of the model to evaluate
        eval_dataset: The evaluation dataset
        eval_config: Evaluation configuration
        output_dir: Directory to save results

    Returns:
        ModelResponses object with all results
    """
    start_time = time.time()

    # Load the model using unified resolution - all models go through the same resolver
    model, tokenizer, metadata = load_model_from_spec(model_spec.source_path)

    # Determine model type and ID based on source type
    if model_spec.source_type in ["suite_experiment", "run_path"]:
        model_type = "trained"
    elif model_spec.source_type == "model_config":
        model_type = "config"
    elif model_spec.source_type == "base_model":
        model_type = "base"
    else:
        raise ValueError(f"Unknown source type: {model_spec.source_type}")
    model_id = f"{model_spec.source_path.replace('/', '_')}"

    # Extract prompts
    prompts = [p.prompt for p in eval_dataset.prompts]

    # Run inference
    display_name = model_spec.display_name or model_spec.source_path
    responses = run_inference_on_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=eval_config.get("max_new_tokens", 4096),
        temperature=eval_config.get("temperature", 0.7),
        top_p=eval_config.get("top_p", 0.9),
        model_name=display_name,
        model_type=metadata["model_type"],
        show_progress=False,  # Disable inner progress bar to avoid conflicts with outer benchmark progress
    )

    # Create response pairs
    response_pairs = []
    for prompt_obj, response in zip(eval_dataset.prompts, responses):
        response_pairs.append(
            ModelResponsePair(
                prompt=prompt_obj.prompt,
                response=response,
                metadata={
                    **prompt_obj.metadata,
                    "prompt_id": prompt_obj.get_id(),  # Add prompt ID to metadata
                },
            )
        )

    # Calculate metrics
    total_time = time.time() - start_time
    successful_responses = sum(1 for r in responses if not r.startswith("ERROR:"))
    avg_response_length = (
        sum(len(r.split()) for r in responses) / len(responses) if responses else 0
    )

    # Create ModelResponses object
    model_responses = ModelResponses(
        model_id=model_id,
        model_path=model_spec.source_path,
        model_type=model_type,
        run_metadata=metadata.get("run_metadata"),
        eval_dataset_name=eval_dataset.name,
        eval_config=eval_config,
        base_model=metadata["base_model"],
        peft_applied=metadata["peft_applied"],
        peft_config=metadata["peft_config"],
        responses=response_pairs,
        total_prompts=len(prompts),
        successful_responses=successful_responses,
        average_response_length=avg_response_length,
        total_time_seconds=total_time,
    )

    # Save results
    output_path = Path(output_dir) / f"{model_id}_responses.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(model_responses.model_dump(), f, indent=2, default=str)

    print(f"\nSaved responses to: {output_path}")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print(f"Successful responses: {successful_responses}/{len(prompts)}")

    # Clean up GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return model_responses


def expand_suite_to_model_specs(suite_name: str) -> List[ModelSpec]:
    """
    Expand a suite name to a list of model specifications for all experiments in the suite.

    Args:
        suite_name: Name of the experiment suite

    Returns:
        List of ModelSpec objects for each experiment in the suite
    """
    print(f"\nExpanding suite: {suite_name}")

    suite_dir = Path("experiments/suites") / suite_name
    if not suite_dir.exists():
        raise ValueError(f"Suite not found: {suite_name}")

    # Load suite summary
    summary_path = suite_dir / "suite_summary.json"
    if not summary_path.exists():
        raise ValueError(f"Suite summary not found: {summary_path}")

    with open(summary_path, "r") as f:
        suite_summary = json.load(f)

    # Extract completed experiments
    model_specs = []
    for exp_id, exp_info in suite_summary.get("experiments", {}).items():
        if exp_info.get("status") == "completed" and exp_info.get("final_model_path"):
            model_path = exp_info["final_model_path"]
            # Convert to relative path format expected by find_run
            if model_path.startswith("experiments/"):
                model_path = model_path.replace("experiments/", "")

            model_specs.append(
                ModelSpec(
                    source_type="suite_experiment",
                    source_path=model_path,
                    display_name=f"{suite_name}/{exp_id}",
                )
            )
            print(f"  - Added experiment: {exp_id} -> {model_path}")

    print(f"  Total experiments found: {len(model_specs)}")

    return model_specs
