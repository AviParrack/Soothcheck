import argparse
import os
import torch
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from probity.probes import BaseProbe

from probity.visualisation.token_highlight import generate_token_visualization

from probity.evaluation.batch_evaluator import OptimizedBatchProbeEvaluator
from probity.visualisation.probe_performance import (
    create_probe_type_plots,
    create_performance_plots
)
from probity.utils.dataset_loading import load_lie_truth_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained probes')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--eval_dataset_dir', type=str, required=True)
    parser.add_argument('--probe_dir', type=str, required=True)
    parser.add_argument('--results_save_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for activation collection')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load evaluation dataset
    print(f"Loading evaluation dataset from {args.eval_dataset_dir}")
    dataset = load_lie_truth_dataset(args.eval_dataset_dir)
    
    # Extract texts and labels
    texts = [ex.text for ex in dataset.examples]
    labels = [ex.label for ex in dataset.examples]
    
    # Create results directory
    results_dir = Path(args.results_save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create optimized evaluator
    evaluator = OptimizedBatchProbeEvaluator(args.model_name, args.device)
    
    # Load all probes
    probe_configs = {}
    probe_dir = Path(args.probe_dir)
    
    print("Loading all probes...")
    for probe_type_dir in probe_dir.iterdir():
        if not probe_type_dir.is_dir():
            continue

        if probe_type_dir.name.startswith('.') or probe_type_dir.name in ['__pycache__', '.ipynb_checkpoints']:
            continue
        
        probe_type = probe_type_dir.name
        probe_files = sorted(probe_type_dir.glob("layer_*_probe.json"))

        if not probe_files:
            print(f"No probe files found in {probe_type_dir}, skipping...")
            continue
        
        for probe_file in probe_files:
            # Extract layer number
            layer = int(probe_file.stem.split('_')[1])
            
            # Load probe
            probe = BaseProbe.load_json(str(probe_file))
            probe_configs[(layer, probe_type)] = probe
    
    print(f"Loaded {len(probe_configs)} probes")
    
    # Evaluate all probes efficiently
    print("Running optimized evaluation...")
    all_results = evaluator.evaluate_all_probes(texts, labels, probe_configs)
    
    # Reorganize results and save
    print("Saving results...")
    results_by_type = {}
    
    for (layer, probe_type), result in all_results.items():
        if probe_type not in results_by_type:
            results_by_type[probe_type] = {}
        results_by_type[probe_type][layer] = result
        
        # Save individual layer results
        layer_dir = results_dir / probe_type / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        with open(layer_dir / 'full_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        # Generate visualization
        generate_token_visualization(result['token_details'], 
                                  layer_dir / 'token_visualization.html')
    
    # Save probe type summaries and create plots
    for probe_type, probe_results in results_by_type.items():
        probe_type_dir = results_dir / probe_type
        probe_type_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all layers results
        with open(probe_type_dir / 'all_layers_results.json', 'w') as f:
            json.dump(probe_results, f, indent=2)
        
        # Save CSV
        csv_data = []
        for layer, result in probe_results.items():
            row = {'layer': layer}
            row.update(result['metrics'])
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(probe_type_dir / 'metrics.csv', index=False)
        
        # Create plots
        create_probe_type_plots(probe_results, probe_type_dir, probe_type)
    
    # Save final combined results
    json_results = {f"{layer}_{probe_type}": result for (layer, probe_type), result in all_results.items()}
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # Create overall visualization plots
    create_performance_plots(all_results, results_dir)
    print(f"\nEvaluation complete. Results saved to {results_dir}")
    
    # Save best configurations
    best_configs = []
    probe_types = set(probe_type for _, probe_type in all_results.keys())
    
    for probe_type in probe_types:
        best_layer = -1
        best_auroc = -1
        
        for (layer, pt), result in all_results.items():
            if pt == probe_type:
                auroc = result['metrics']['auroc']
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_layer = layer
        
        best_configs.append({
            'probe_type': probe_type,
            'layer': best_layer,
            'auroc': best_auroc
        })
        print(f"{probe_type}: Layer {best_layer} (AUROC: {best_auroc:.4f})")
    
    with open(results_dir / 'best_configurations.json', 'w') as f:
        json.dump(best_configs, f, indent=2)


if __name__ == "__main__":
    main()