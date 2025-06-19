# Model comparison and judging functionality for evaluating relative model performance
# Supports LLM-based and human judges, with interruptible judging sessions
# and ELO score calculation from win/loss statistics

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
import itertools
import hashlib

import matplotlib.pyplot as plt
import numpy as np

from src.config_models.eval_configs import (
    ModelResponses,
    ComparisonPair,
    JudgmentResult,
    JudgingSession,
    ModelWinStats,
    ComparisonResults,
)
from src.utils.litellm_logger import litellm_completion


def load_all_model_responses(dataset_name: str) -> Dict[str, ModelResponses]:
    """
    Load all model responses for a given dataset.

    Args:
        dataset_name: Name of the evaluation dataset

    Returns:
        Dictionary mapping model_id to ModelResponses
    """
    responses_dir = Path("evals/model_responses") / dataset_name

    if not responses_dir.exists():
        raise ValueError(f"No responses found for dataset: {dataset_name}")

    model_responses = {}

    # Load all response files
    for response_file in responses_dir.glob("*_responses.json"):
        with open(response_file, "r") as f:
            data = json.load(f)

        # Create ModelResponses object from the data
        responses = ModelResponses(**data)
        model_responses[responses.model_id] = responses

    print(f"Loaded responses from {len(model_responses)} models")

    return model_responses


def validate_responses_consistency(
    model_responses: Dict[str, ModelResponses],
) -> Tuple[bool, List[str]]:
    """
    Check that all models answered the same prompts.

    Returns:
        Tuple of (is_consistent, list_of_warnings)
    """
    warnings = []

    if len(model_responses) < 2:
        return True, ["Only one model found, nothing to compare"]

    # Get prompt sets from each model
    prompt_sets = {}
    for model_id, responses in model_responses.items():
        # Extract prompts and their IDs
        prompts_with_ids = set()
        for response_pair in responses.responses:
            prompt_id = response_pair.metadata.get("prompt_id")
            if not prompt_id:
                # Generate ID from prompt text if not present
                prompt_id = hashlib.md5(response_pair.prompt.encode()).hexdigest()[:12]
            prompts_with_ids.add((prompt_id, response_pair.prompt))
        prompt_sets[model_id] = prompts_with_ids

    # Find the reference set (first model)
    reference_model = list(model_responses.keys())[0]
    reference_prompts = prompt_sets[reference_model]

    # Check consistency
    is_consistent = True
    for model_id, prompts in prompt_sets.items():
        if model_id == reference_model:
            continue

        missing_in_model = reference_prompts - prompts
        extra_in_model = prompts - reference_prompts

        if missing_in_model:
            is_consistent = False
            warnings.append(f"Model {model_id} missing {len(missing_in_model)} prompts")

        if extra_in_model:
            is_consistent = False
            warnings.append(f"Model {model_id} has {len(extra_in_model)} extra prompts")

    return is_consistent, warnings


def create_comparison_pairs(
    model_responses: Dict[str, ModelResponses],
) -> List[ComparisonPair]:
    """
    Create all possible comparison pairs between models for each prompt.

    Args:
        model_responses: Dictionary of model responses

    Returns:
        List of comparison pairs
    """
    pairs = []
    model_ids = list(model_responses.keys())

    # Get all unique prompts (using the first model as reference)
    first_model = model_responses[model_ids[0]]
    prompts_map = {}
    for response_pair in first_model.responses:
        prompt_id = response_pair.metadata.get("prompt_id")
        if not prompt_id:
            # Generate ID from prompt text if not present
            prompt_id = hashlib.md5(response_pair.prompt.encode()).hexdigest()[:12]
        prompts_map[prompt_id] = response_pair.prompt

    # For each prompt, create pairs between all models
    for prompt_id, prompt_text in prompts_map.items():
        # Get all pairwise combinations of models
        for model_a, model_b in itertools.combinations(model_ids, 2):
            # Find responses for this prompt from both models
            response_a = None
            response_b = None

            for resp in model_responses[model_a].responses:
                resp_prompt_id = resp.metadata.get("prompt_id")
                if not resp_prompt_id:
                    # Generate ID from prompt text if not present
                    resp_prompt_id = hashlib.md5(resp.prompt.encode()).hexdigest()[:12]

                if resp_prompt_id == prompt_id:
                    response_a = resp.response
                    break

            for resp in model_responses[model_b].responses:
                resp_prompt_id = resp.metadata.get("prompt_id")
                if not resp_prompt_id:
                    # Generate ID from prompt text if not present
                    resp_prompt_id = hashlib.md5(resp.prompt.encode()).hexdigest()[:12]

                if resp_prompt_id == prompt_id:
                    response_b = resp.response
                    break

            if response_a and response_b:
                pairs.append(
                    ComparisonPair(
                        prompt_id=prompt_id,
                        prompt=prompt_text,
                        model_a_id=model_a,
                        model_b_id=model_b,
                        response_a=response_a,
                        response_b=response_b,
                    )
                )

    return pairs


def judge_with_gemini(pair: ComparisonPair) -> JudgmentResult:
    """
    Judge a comparison pair using Gemini 2.0 Pro.

    Args:
        pair: The comparison pair to judge

    Returns:
        JudgmentResult with the judgment
    """
    # Randomly assign responses to A/B to avoid position bias
    if random.random() < 0.5:
        response_a = pair.response_a
        response_b = pair.response_b
        model_a_actual = pair.model_a_id
        model_b_actual = pair.model_b_id
    else:
        response_a = pair.response_b
        response_b = pair.response_a
        model_a_actual = pair.model_b_id
        model_b_actual = pair.model_a_id

    # Create the judgment prompt
    prompt = f"""You are an expert judge evaluating two AI model responses to the same prompt.

**Original Prompt:**
{pair.prompt}

**Response A:**
{response_a}

**Response B:**
{response_b}

Please evaluate which response is better overall. Consider factors like:
- Accuracy and correctness
- Helpfulness and relevance
- Clarity and coherence
- Completeness
- Following instructions

You must choose one of: A, B, or tie

Use the judge_response function to provide your judgment."""

    # Define the tool for structured output
    tools = [
        {
            "type": "function",
            "function": {
                "name": "judge_response",
                "description": "Provide judgment on which response is better",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "winner": {
                            "type": "string",
                            "enum": ["A", "B", "tie"],
                            "description": "Which response is better: A, B, or tie",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief explanation of your judgment",
                        },
                    },
                    "required": ["winner", "explanation"],
                },
            },
        }
    ]

    # Call Gemini with tool use
    try:
        response = litellm_completion(
            model="gemini/gemini-2.0-pro-05-02",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="required",
            temperature=0.1,  # Low temperature for more consistent judgments
        )

        # Extract the function call
        tool_call = response.choices[0].message.tool_calls[0]
        judgment_data = json.loads(tool_call.function.arguments)

        winner_str = judgment_data["winner"]
        explanation = judgment_data["explanation"]

        # Map back to actual model IDs
        if winner_str == "A":
            winner = "A" if model_a_actual == pair.model_a_id else "B"
        elif winner_str == "B":
            winner = "B" if model_b_actual == pair.model_b_id else "A"
        else:
            winner = "tie"

    except Exception as e:
        print(f"Error in Gemini judgment: {e}")
        # Default to tie on error
        winner = "tie"
        explanation = f"Error during judgment: {str(e)}"

    # Create judgment result
    comparison_id = JudgingSession._get_comparison_id(pair)

    return JudgmentResult(
        comparison_id=comparison_id,
        prompt_id=pair.prompt_id,
        model_a_id=pair.model_a_id,
        model_b_id=pair.model_b_id,
        winner=winner,
        judge_type="gemini-2.0-pro",
        judge_explanation=explanation,
    )


def judge_with_human(pair: ComparisonPair) -> JudgmentResult:
    """
    Judge a comparison pair using human input via CLI.

    Args:
        pair: The comparison pair to judge

    Returns:
        JudgmentResult with the judgment
    """
    # Clear screen for better readability
    print("\n" * 3)
    print("=" * 80)
    print("HUMAN JUDGMENT REQUIRED")
    print("=" * 80)

    print(f"\n**Original Prompt:**\n{pair.prompt}\n")

    # Randomly assign responses to A/B
    if random.random() < 0.5:
        response_a = pair.response_a
        response_b = pair.response_b
        model_a_actual = pair.model_a_id
        model_b_actual = pair.model_b_id
    else:
        response_a = pair.response_b
        response_b = pair.response_a
        model_a_actual = pair.model_b_id
        model_b_actual = pair.model_a_id

    print(f"\n{'='*40} RESPONSE A {'='*40}")
    print(response_a)

    print(f"\n{'='*40} RESPONSE B {'='*40}")
    print(response_b)

    print("\n" + "=" * 80)

    # Get judgment
    while True:
        judgment = input("\nWhich response is better? (A/B/tie): ").strip().lower()
        if judgment in ["a", "b", "tie"]:
            break
        print("Invalid input. Please enter A, B, or tie.")

    # Get explanation
    explanation = input("Brief explanation (optional, press Enter to skip): ").strip()
    if not explanation:
        explanation = "Human judgment without explanation"

    # Map judgment to winner
    if judgment == "a":
        winner = "A" if model_a_actual == pair.model_a_id else "B"
    elif judgment == "b":
        winner = "B" if model_b_actual == pair.model_b_id else "A"
    else:
        winner = "tie"

    comparison_id = JudgingSession._get_comparison_id(pair)

    return JudgmentResult(
        comparison_id=comparison_id,
        prompt_id=pair.prompt_id,
        model_a_id=pair.model_a_id,
        model_b_id=pair.model_b_id,
        winner=winner,
        judge_type="human",
        judge_explanation=explanation,
    )


def load_or_create_session(
    dataset_name: str, judge_type: str, total_comparisons: int
) -> JudgingSession:
    """
    Load existing judging session or create a new one.

    Args:
        dataset_name: Name of the dataset
        judge_type: Type of judge (e.g., "human", "gemini-2.0-pro")
        total_comparisons: Total number of comparisons to judge

    Returns:
        JudgingSession object
    """
    session_dir = Path("evals/grading") / dataset_name
    session_dir.mkdir(parents=True, exist_ok=True)

    session_file = session_dir / f"session_{judge_type}.json"

    if session_file.exists():
        # Load existing session
        with open(session_file, "r") as f:
            data = json.load(f)
        session = JudgingSession(**data)
        print(
            f"Resuming session: {session.completed_comparisons}/{session.total_comparisons} completed"
        )
    else:
        # Create new session
        session_id = (
            f"{dataset_name}_{judge_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        session = JudgingSession(
            dataset_name=dataset_name,
            session_id=session_id,
            judge_type=judge_type,
            total_comparisons=total_comparisons,
            completed_comparisons=0,
        )
        print(f"Starting new judging session with {total_comparisons} comparisons")

    return session


def save_session(session: JudgingSession):
    """Save judging session to disk."""
    session_dir = Path("evals/grading") / session.dataset_name
    session_dir.mkdir(parents=True, exist_ok=True)

    session_file = session_dir / f"session_{session.judge_type}.json"

    # Update timestamps
    session.last_updated = datetime.now()

    with open(session_file, "w") as f:
        json.dump(session.model_dump(), f, indent=2, default=str)


def calculate_elo_ratings(
    judgments: List[JudgmentResult], initial_rating: float = 1500.0
) -> Dict[str, float]:
    """
    Calculate ELO ratings from judgment results.

    Args:
        judgments: List of judgment results
        initial_rating: Initial ELO rating for all models

    Returns:
        Dictionary mapping model_id to ELO rating
    """
    # Initialize ratings
    ratings = {}

    # Get all unique model IDs
    model_ids = set()
    for judgment in judgments:
        model_ids.add(judgment.model_a_id)
        model_ids.add(judgment.model_b_id)

    for model_id in model_ids:
        ratings[model_id] = initial_rating

    # Process judgments in order
    k_factor = 32  # Standard K-factor for ELO

    for judgment in judgments:
        model_a = judgment.model_a_id
        model_b = judgment.model_b_id

        # Current ratings
        rating_a = ratings[model_a]
        rating_b = ratings[model_b]

        # Expected scores
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

        # Actual scores
        if judgment.winner == "A":
            score_a, score_b = 1.0, 0.0
        elif judgment.winner == "B":
            score_a, score_b = 0.0, 1.0
        else:  # tie
            score_a, score_b = 0.5, 0.5

        # Update ratings
        ratings[model_a] = rating_a + k_factor * (score_a - expected_a)
        ratings[model_b] = rating_b + k_factor * (score_b - expected_b)

    return ratings


def create_elo_chart(comparison_results: ComparisonResults, output_path: Path):
    """
    Create a bar chart of ELO ratings.

    Args:
        comparison_results: The comparison results with ELO ratings
        output_path: Path to save the chart
    """
    if not comparison_results.elo_ratings:
        print("No ELO ratings to plot")
        return

    # Sort models by ELO rating
    sorted_models = sorted(
        comparison_results.elo_ratings.items(), key=lambda x: x[1], reverse=True
    )

    model_names = [m[0] for m in sorted_models]
    elo_scores = [m[1] for m in sorted_models]

    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(model_names)), elo_scores)

    # Color bars based on score
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Customize the plot
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("ELO Rating", fontsize=12)
    plt.title(f"Model ELO Ratings - {comparison_results.dataset_name}", fontsize=14)
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")

    # Add value labels on bars
    for i, (name, score) in enumerate(sorted_models):
        plt.text(i, score + 10, f"{score:.0f}", ha="center", va="bottom")

    # Add horizontal line at initial rating
    plt.axhline(y=1500, color="gray", linestyle="--", alpha=0.7, label="Initial Rating")

    # Add grid
    plt.grid(axis="y", alpha=0.3)

    # Add legend
    plt.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"ELO chart saved to: {output_path}")


def run_comparison(
    dataset_name: str,
    judge_type: str = "gemini-2.0-pro",
    max_comparisons: Optional[int] = None,
) -> ComparisonResults:
    """
    Run model comparison for a dataset.

    Args:
        dataset_name: Name of the evaluation dataset
        judge_type: Type of judge to use ("human" or "gemini-2.0-pro")
        max_comparisons: Maximum number of comparisons to run (for testing)

    Returns:
        ComparisonResults object
    """
    # Load all model responses
    model_responses = load_all_model_responses(dataset_name)

    # Validate consistency
    is_consistent, warnings = validate_responses_consistency(model_responses)

    if warnings:
        print("\nWarnings found:")
        for warning in warnings:
            print(f"  - {warning}")

        if not is_consistent:
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != "y":
                print("Aborting comparison")
                return None

    # Create comparison pairs
    all_pairs = create_comparison_pairs(model_responses)

    if max_comparisons:
        all_pairs = all_pairs[:max_comparisons]

    print(f"\nTotal comparison pairs: {len(all_pairs)}")

    # Load or create session
    session = load_or_create_session(dataset_name, judge_type, len(all_pairs))

    # Get remaining comparisons
    remaining_pairs = session.get_remaining_comparisons(all_pairs)

    if not remaining_pairs:
        print("All comparisons already completed!")
    else:
        print(f"Comparisons remaining: {len(remaining_pairs)}")

        # Select judge function
        if judge_type == "human":
            judge_fn = judge_with_human
        elif judge_type == "gemini-2.0-pro":
            judge_fn = judge_with_gemini
        else:
            raise ValueError(f"Unknown judge type: {judge_type}")

        # Run comparisons
        try:
            for i, pair in enumerate(remaining_pairs):
                print(
                    f"\nComparison {session.completed_comparisons + 1}/{session.total_comparisons}"
                )

                # Judge the pair
                judgment = judge_fn(pair)

                # Add to session
                session.judgments.append(judgment)
                session.completed_comparisons += 1

                # Save after each judgment (for interruption safety)
                save_session(session)

                # Show progress
                print(f"Winner: {judgment.winner}")
                if judgment.judge_explanation:
                    print(f"Explanation: {judgment.judge_explanation}")

        except KeyboardInterrupt:
            print("\n\nInterrupted! Progress saved.")
            save_session(session)
            return None

    # Calculate final statistics
    print("\n" + "=" * 80)
    print("CALCULATING FINAL RESULTS")
    print("=" * 80)

    # Initialize model stats
    model_stats = {}
    for model_id in model_responses.keys():
        model_stats[model_id] = ModelWinStats(model_id=model_id)

    # Process judgments
    for judgment in session.judgments:
        # Update stats for model A
        stats_a = model_stats[judgment.model_a_id]
        stats_a.total_comparisons += 1

        if judgment.winner == "A":
            stats_a.wins += 1
        elif judgment.winner == "B":
            stats_a.losses += 1
        else:
            stats_a.ties += 1

        # Update stats for model B
        stats_b = model_stats[judgment.model_b_id]
        stats_b.total_comparisons += 1

        if judgment.winner == "B":
            stats_b.wins += 1
        elif judgment.winner == "A":
            stats_b.losses += 1
        else:
            stats_b.ties += 1

    # Calculate ELO ratings
    elo_ratings = calculate_elo_ratings(session.judgments)

    # Update model stats with ELO
    for model_id, rating in elo_ratings.items():
        model_stats[model_id].elo_rating = rating

    # Create final results
    comparison_results = ComparisonResults(
        dataset_name=dataset_name,
        total_models=len(model_responses),
        total_comparisons=len(session.judgments),
        judge_types=[judge_type],
        model_stats=model_stats,
        all_judgments=session.judgments,
        elo_ratings=elo_ratings,
    )

    # Save final results
    results_dir = Path("evals/grading") / dataset_name
    results_file = results_dir / f"comparison_results_{judge_type}.json"

    with open(results_file, "w") as f:
        json.dump(comparison_results.model_dump(), f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Create ELO chart
    chart_path = results_dir / f"elo_chart_{judge_type}.png"
    create_elo_chart(comparison_results, chart_path)

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RANKINGS (by ELO)")
    print("=" * 80)

    sorted_by_elo = sorted(
        model_stats.items(), key=lambda x: x[1].elo_rating or 0, reverse=True
    )

    for rank, (model_id, stats) in enumerate(sorted_by_elo, 1):
        print(f"\n{rank}. {model_id}")
        print(
            f"   ELO: {stats.elo_rating:.0f}"
            if stats.elo_rating is not None
            else "   ELO: N/A"
        )
        print(f"   Wins: {stats.wins}, Losses: {stats.losses}, Ties: {stats.ties}")
        print(f"   Win Rate: {stats.win_rate:.1%}")

    return comparison_results
