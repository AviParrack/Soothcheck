#!/usr/bin/env python3
"""Script to find the best performing layer from NTML probe metrics."""

import json
from pathlib import Path
import argparse

def find_best_layer(probe_dir: str) -> None:
    """Find the best performing layer based on AUROC."""
    probe_dir = Path(probe_dir)
    
    # Store layer metrics
    layer_metrics = {}
    
    # Read all metrics files
    for metrics_file in probe_dir.glob("ntml_binary_4T1L_500samples_layer_*_metrics.json"):
        # Extract layer number
        layer = int(metrics_file.stem.split('_')[-2])
        
        # Read metrics
        with open(metrics_file) as f:
            metrics = json.load(f)
            layer_metrics[layer] = metrics
    
    # Sort layers by AUROC
    sorted_layers = sorted(
        layer_metrics.items(),
        key=lambda x: x[1]['auroc'],
        reverse=True
    )
    
    # Print top 5 layers
    print("\nTop 5 layers by AUROC:")
    print("=" * 50)
    print(f"{'Layer':>6} | {'AUROC':>10} | {'Accuracy':>10} | {'F1':>10}")
    print("-" * 50)
    
    for layer, metrics in sorted_layers[:5]:
        print(f"{layer:6d} | {metrics['auroc']:10.4f} | {metrics['accuracy']:10.4f} | {metrics['f1']:10.4f}")

def main():
    parser = argparse.ArgumentParser(description='Find best performing NTML probe layer')
    parser.add_argument('probe_dir', help='Directory containing probe metrics files')
    args = parser.parse_args()
    
    find_best_layer(args.probe_dir)

if __name__ == '__main__':
    main() 