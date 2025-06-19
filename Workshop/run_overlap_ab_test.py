#!/usr/bin/env python3
"""
A/B Test: Overlapping Chunker Impact on Model Performance

This script trains the same model twice:
- Condition A: No overlap (baseline)
- Condition B: 15% overlap (experimental)

Then compares training metrics and evaluation performance.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import time

def create_ab_test_configs():
    """Create configuration files for both test conditions"""
    
    # Base dataset config template
    base_build_config = {
        "output_suffix": "_sft",
        "validation_split_fraction": 0.2,
        "tokenizer_name_or_path": "distilbert/distilgpt2",
        "max_length": 150,  # Force chunking
        "enable_chunking": True,
        "chunk_at_markdown_headers": True,
        "prompt_format": "text",
        "data_pipelines": [
            {
                "name": "basic",
                "sources": [
                    {
                        "folder": "corpus"
                    }
                ]
            }
        ]
    }
    
    # Condition A: No overlap
    build_config_a = base_build_config.copy()
    build_config_a["chunk_overlap_ratio"] = 0.0
    
    config_a = {
        "split_seed": 42,
        "builds": {
            "sft": build_config_a
        }
    }
    
    # Condition B: With overlap 
    build_config_b = base_build_config.copy()
    build_config_b["chunk_overlap_ratio"] = 0.15
    
    config_b = {
        "split_seed": 42,
        "builds": {
            "sft": build_config_b
        }
    }
    
    # Save configs
    os.makedirs("configs/data", exist_ok=True)
    
    with open("configs/data/ab_test_no_overlap.json", "w") as f:
        json.dump(config_a, f, indent=2)
        
    with open("configs/data/ab_test_with_overlap.json", "w") as f:
        json.dump(config_b, f, indent=2)
    
    print("✓ Created A/B test configurations")
    return config_a, config_b

def setup_test_dataset():
    """Copy example dataset to our test directory"""
    source_dir = Path("datasets/raw/example_dataset")
    target_dir = Path("datasets/raw/overlap_ab_test")
    
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    # Create target structure
    corpus_dir = target_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    
    # Copy files
    for file in source_dir.glob("*.mdx"):
        shutil.copy2(file, corpus_dir)
    
    print(f"✓ Copied dataset from {source_dir} to {target_dir}")
    print(f"✓ Dataset files: {list(corpus_dir.glob('*.mdx'))}")

def run_single_experiment(condition_name: str, config_file: str) -> Dict[str, Any]:
    """Run a single training experiment and return metrics"""
    
    print(f"\n{'='*60}")
    print(f"RUNNING CONDITION: {condition_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Step 1: Prepare dataset
    dataset_name = config_file.replace('.json', '').replace('ab_test_', '')
    print(f"Step 1: Preparing dataset '{dataset_name}'...")
    
    prepare_cmd = f"python -m scripts.prepare_dataset {config_file.replace('.json', '')} overlap_ab_test sft"
    print(f"Dataset preparation command: {prepare_cmd}")
    
    exit_code = os.system(prepare_cmd)
    if exit_code != 0:
        print(f"❌ Dataset preparation failed for {condition_name}")
        return {"status": "failed", "stage": "dataset_prep", "exit_code": exit_code}
    
    print(f"✓ Dataset prepared successfully")
    
    # Step 2: Run training 
    experiment_name = f"ab_test_{condition_name}_{int(time.time())}"
    
    train_cmd = f"python -m scripts.train_pipeline sft_cpu_compatible --dataset overlap_ab_test --model distilgpt2 --run_name {experiment_name}"
    
    print(f"Step 2: Training model...")
    print(f"Training command: {train_cmd}")
    
    # Run training
    exit_code = os.system(train_cmd)
    
    if exit_code != 0:
        print(f"❌ Training failed for {condition_name}")
        return {"status": "failed", "stage": "training", "exit_code": exit_code}
    
    # Extract metrics from experiment directory
    exp_dir = Path("experiments") / experiment_name
    metrics = extract_training_metrics(exp_dir)
    
    metrics.update({
        "condition": condition_name,
        "experiment_name": experiment_name,
        "training_time_seconds": time.time() - start_time,
        "status": "success"
    })
    
    print(f"✓ Completed {condition_name} in {metrics['training_time_seconds']:.1f}s")
    return metrics

def extract_training_metrics(exp_dir: Path) -> Dict[str, Any]:
    """Extract key metrics from experiment directory"""
    metrics = {}
    
    # 1. Final training metrics from trainer_state.json
    trainer_state_path = exp_dir / "trainer_state.json" 
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)
            
        if "log_history" in trainer_state and trainer_state["log_history"]:
            final_log = trainer_state["log_history"][-1]
            metrics.update({
                "final_train_loss": final_log.get("train_loss"),
                "final_eval_loss": final_log.get("eval_loss"),
                "final_epoch": final_log.get("epoch"),
                "total_steps": trainer_state.get("global_step", 0)
            })
    
    # 2. All results from all_results.json
    results_path = exp_dir / "all_results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
            metrics.update(all_results)
    
    # 3. Dataset info from processed dataset
    processed_dir = Path("datasets/processed") / exp_dir.name
    if processed_dir.exists():
        dataset_files = list(processed_dir.glob("*.json"))
        if dataset_files:
            with open(dataset_files[0]) as f:
                dataset_info = json.load(f)
                if isinstance(dataset_info, list) and dataset_info:
                    metrics.update({
                        "num_training_samples": len(dataset_info),
                        "avg_sample_length": sum(len(item.get("text", "")) for item in dataset_info) / len(dataset_info)
                    })
    
    return metrics

def analyze_results(results_a: Dict, results_b: Dict):
    """Compare results between conditions"""
    
    print(f"\n{'='*80}")
    print("A/B TEST RESULTS ANALYSIS")
    print(f"{'='*80}")
    
    # Training metrics comparison
    print("\n📊 TRAINING METRICS COMPARISON")
    print("-" * 50)
    
    metrics_to_compare = [
        ("final_train_loss", "Final Training Loss", "lower_better"),
        ("final_eval_loss", "Final Eval Loss", "lower_better"), 
        ("total_steps", "Training Steps", "neutral"),
        ("training_time_seconds", "Training Time (s)", "lower_better"),
        ("num_training_samples", "Training Samples", "higher_better"),
        ("avg_sample_length", "Avg Sample Length", "neutral")
    ]
    
    for metric_key, metric_name, direction in metrics_to_compare:
        val_a = results_a.get(metric_key, "N/A")
        val_b = results_b.get(metric_key, "N/A")
        
        if val_a != "N/A" and val_b != "N/A":
            diff = val_b - val_a
            pct_change = (diff / val_a) * 100 if val_a != 0 else 0
            
            if direction == "lower_better":
                winner = "B (Overlap)" if val_b < val_a else "A (No Overlap)"
                symbol = "📉" if val_b < val_a else "📈"
            elif direction == "higher_better":
                winner = "B (Overlap)" if val_b > val_a else "A (No Overlap)"
                symbol = "📈" if val_b > val_a else "📉"
            else:
                winner = "Neutral"
                symbol = "📊"
            
            print(f"{symbol} {metric_name}:")
            print(f"   No Overlap (A): {val_a:.4f}" if isinstance(val_a, float) else f"   No Overlap (A): {val_a}")
            print(f"   With Overlap (B): {val_b:.4f}" if isinstance(val_b, float) else f"   With Overlap (B): {val_b}")
            if isinstance(diff, (int, float)):
                print(f"   Difference: {diff:+.4f} ({pct_change:+.1f}%)")
            print(f"   Winner: {winner}")
            print()
    
    # Summary
    print("\n🏆 OVERALL ASSESSMENT")
    print("-" * 30)
    
    # Check key improvement metrics
    improvements = []
    
    if results_a.get("final_train_loss") and results_b.get("final_train_loss"):
        if results_b["final_train_loss"] < results_a["final_train_loss"]:
            improvements.append("Lower training loss")
    
    if results_a.get("final_eval_loss") and results_b.get("final_eval_loss"):
        if results_b["final_eval_loss"] < results_a["final_eval_loss"]:
            improvements.append("Lower evaluation loss")
    
    if results_a.get("num_training_samples") and results_b.get("num_training_samples"):
        if results_b["num_training_samples"] > results_a["num_training_samples"]:
            improvements.append("More training samples (better chunking)")
    
    if improvements:
        print("✅ Overlap chunking showed improvements:")
        for improvement in improvements:
            print(f"   • {improvement}")
    else:
        print("❓ No clear improvements detected with overlap chunking")
        print("   This could indicate:")
        print("   • Dataset too small to show benefits")
        print("   • Overlap ratio needs tuning") 
        print("   • Task doesn't benefit from overlap")
    
    # Save detailed results
    results_summary = {
        "timestamp": time.time(),
        "condition_a_no_overlap": results_a,
        "condition_b_with_overlap": results_b,
        "improvements_detected": improvements,
        "conclusion": "overlap_beneficial" if improvements else "inconclusive"
    }
    
    with open("ab_test_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: ab_test_results.json")

def main():
    """Run the complete A/B test"""
    
    print("🧪 OVERLAP CHUNKER A/B TEST")
    print("="*50)
    print("Testing hypothesis: Overlapping chunks improve model performance")
    print("Model: distilbert/distilgpt2 (82M parameters)")
    print("Dataset: example_dataset (ML + AI safety content)")
    print()
    
    # Setup
    print("🔧 SETUP PHASE")
    setup_test_dataset()
    config_a, config_b = create_ab_test_configs()
    
    # Run experiments
    print("\n🚀 TRAINING PHASE")
    
    # Condition A: No overlap
    results_a = run_single_experiment("no_overlap", "ab_test_no_overlap")
    
    # Condition B: With overlap
    results_b = run_single_experiment("with_overlap", "ab_test_with_overlap")
    
    # Analysis
    print("\n📊 ANALYSIS PHASE")
    if results_a["status"] == "success" and results_b["status"] == "success":
        analyze_results(results_a, results_b)
    else:
        print("❌ Cannot analyze results - one or both experiments failed")
        print(f"Condition A status: {results_a['status']}")
        print(f"Condition B status: {results_b['status']}")
    
    print("\n✅ A/B Test Complete!")

if __name__ == "__main__":
    main() 