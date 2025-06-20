import json
import numpy as np
from pathlib import Path



def create_2t1l_plots(probe_type_dir: Path, probe_type: str):
    """Create 2T1L performance plots for a probe type"""
    import matplotlib.pyplot as plt
    
    # Load the aggregated results
    results_file = probe_type_dir / 'all_layers_2t1l.json'
    if not results_file.exists():
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    layers = sorted([int(l) for l in data['layers'].keys()])
    
    # Extract metrics by layer
    probe_accs = []
    probe_losses = []
    random_accs = []
    random_losses = []
    delta_seps = []
    
    for layer in layers:
        layer_data = data['layers'][str(layer)]
        probe_accs.append(layer_data['probe_performance']['ttol_accuracy'])
        probe_losses.append(layer_data['probe_performance']['avg_ttol_loss'])
        
        if layer_data.get('random_baseline'):
            random_accs.append(layer_data['random_baseline']['ttol_accuracy'])
            random_losses.append(layer_data['random_baseline']['avg_ttol_loss'])
        else:
            random_accs.append(1/3)  # Theoretical random accuracy
            random_losses.append(0.0)  # Theoretical random loss
        
        if layer_data.get('delta_analysis'):
            delta_seps.append(layer_data['delta_analysis']['avg_delta_separation'])
        else:
            delta_seps.append(0.0)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cross-entropy loss
    ax = axes[0, 0]
    ax.plot(layers, probe_losses, 'b-o', label='Probe Loss', linewidth=2)
    ax.plot(layers, random_losses, 'r--', label='Random Baseline', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title(f'{probe_type} - 2T1L Cross-Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax = axes[0, 1]
    ax.plot(layers, probe_accs, 'b-o', label='Probe Accuracy', linewidth=2)
    ax.plot(layers, random_accs, 'r--', label='Random Baseline', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{probe_type} - 2T1L Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Improvement over random
    ax = axes[1, 0]
    acc_improvements = [p - r for p, r in zip(probe_accs, random_accs)]
    loss_improvements = [r - p for p, r in zip(probe_losses, random_losses)]
    ax.plot(layers, acc_improvements, 'g-o', label='Accuracy Improvement', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Improvement over Random')
    ax.set_title(f'{probe_type} - Improvement over Random Baseline')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Delta separation
    ax = axes[1, 1]
    ax.plot(layers, delta_seps, 'm-o', label='Delta Separation', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Lie vs Truth Score Delta')
    ax.set_title(f'{probe_type} - Statement Score Delta Separation')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(probe_type_dir / '2t1l_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved 2T1L plots to {probe_type_dir / '2t1l_performance.png'}")


