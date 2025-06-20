import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional

from probity.analysis.ttol_analysis import (
    extract_three_statements_from_tokens,
    identify_lie_statement_simple,
    analyze_2t1l_from_saved_scores,
    analyze_2t1l_random_baseline,
    extract_assistant_response_from_chat,
    analyze_statement_deltas,
    calculate_delta_statistics
)
from probity.visualisation.ttol_visualisation import create_2t1l_plots

def add_2t1l_analysis_to_saved_results(results_dir: str, dataset_path: str):
    """Add 2T1L analysis to existing saved results"""
    
    # Load original dataset
    with open(dataset_path, 'r') as f:
        raw_data = json.load(f)
    
    # Create lookup using just the model response text (not the full chat template)
    truth_lie_pairs = {}
    for item in raw_data:
        truth_text = item["truth_version"]["model"]  # This is just the assistant response
        truth_lie_pairs[truth_text] = {
            'lie_text': item["lie_version"]["model"],
            'lie_statement': item["lie_statement"],
            'item_id': item['id']
        }
    
    print(f"Loaded {len(truth_lie_pairs)} truth/lie pairs from dataset")
    
    results_path = Path(results_dir)
    all_probe_type_summaries = {}
    
    # Process each probe type
    for probe_type_dir in results_path.iterdir():
        if not probe_type_dir.is_dir() or probe_type_dir.name.startswith('.'):
            continue
        
        probe_type = probe_type_dir.name
        print(f"\nProcessing {probe_type} probes for 2T1L analysis")
        
        probe_type_layer_results = {}
        
        # Process each layer
        for layer_dir in probe_type_dir.iterdir():
            if not layer_dir.is_dir() or not layer_dir.name.startswith('layer_'):
                continue
            
            layer_num = layer_dir.name.split('_')[1]
            sample_scores_file = layer_dir / 'full_results.json'
            
            if not sample_scores_file.exists():
                continue
            
            print(f"  Processing {probe_type} layer {layer_num}")
            
            # Load the saved sample scores
            with open(sample_scores_file, 'r') as f:
                data = json.load(f)

            samples = data["all_samples"]
            ttol_results = []
            random_baseline_results = []
            matches_found = 0
            
            # Create lookup for both truth and lie versions in samples
            sample_lookup = {}
            for sample in samples:
                assistant_response = extract_assistant_response_from_chat(sample['text'])
                if assistant_response:
                    sample_lookup[assistant_response] = sample
            
            # Process each sample
            for sample in samples:
                sample_text = sample['text']
                
                # Extract just the assistant response from the full chat template
                assistant_response = extract_assistant_response_from_chat(sample_text)
                
                # Check if the extracted assistant response matches any truth version
                if assistant_response and assistant_response in truth_lie_pairs:
                    matches_found += 1
                    pair_info = truth_lie_pairs[assistant_response]
                    lie_statement = pair_info['lie_statement']
                    lie_response = pair_info['lie_text']
                    
                    # Find the corresponding lie version sample
                    lie_sample = sample_lookup.get(lie_response)
                    if not lie_sample:
                        continue
                    
                    # Extract statements from both truth and lie versions
                    truth_statements_info = extract_three_statements_from_tokens(
                        sample['tokens'], 
                        sample['token_scores'],
                        sample['text'],
                        lie_statement
                    )
                    
                    lie_statements_info = extract_three_statements_from_tokens(
                        lie_sample['tokens'],
                        lie_sample['token_scores'],
                        lie_sample['text'],
                        lie_statement
                    )
                    
                    if truth_statements_info and lie_statements_info:
                        # Analyze 2T1L loss with probe
                        ttol_analysis = analyze_2t1l_from_saved_scores(
                            truth_statements_info, lie_statement, probe_type
                        )
                        
                        # Analyze delta between truth and lie versions
                        delta_analysis = analyze_statement_deltas(
                            truth_statements_info, lie_statements_info, lie_statement
                        )
                        
                        if ttol_analysis and delta_analysis:
                            # Combine both analyses
                            combined_analysis = {
                                **ttol_analysis,
                                'delta_analysis': delta_analysis,
                                'sample_id': sample['sample_id'],
                                'item_id': pair_info['item_id'],
                                'text': sample_text
                            }
                            ttol_results.append(combined_analysis)
                            
                            # Generate random baseline for this sample
                            random_analysis = analyze_2t1l_random_baseline(truth_statements_info, lie_statement)
                            if random_analysis:
                                random_analysis['sample_id'] = sample['sample_id']
                                random_analysis['item_id'] = pair_info['item_id']
                                random_baseline_results.append(random_analysis)
            
            # Save 2T1L results for this layer
            if ttol_results:
                # Calculate random baseline statistics
                random_baseline_stats = None
                if random_baseline_results:
                    random_baseline_stats = {
                        'avg_ttol_loss': float(np.mean([r['ttol_loss'] for r in random_baseline_results])),
                        'ttol_accuracy': float(np.mean([r['correct_prediction'] for r in random_baseline_results])),
                        'avg_confidence': float(np.mean([r['confidence'] for r in random_baseline_results])),
                        'theoretical_loss': 0.0,  # Should be ~0 after subtracting log(3)
                        'theoretical_accuracy': 1.0/3.0,  # Should be ~33.33%
                        'results': random_baseline_results
                    }
                
                # Calculate delta statistics
                delta_stats = calculate_delta_statistics(ttol_results)
                
                ttol_summary = {
                    'probe_type': probe_type,
                    'layer': int(layer_num),
                    'num_examples': len(ttol_results),
                    'probe_performance': {
                        'avg_ttol_loss': float(np.mean([r['ttol_loss'] for r in ttol_results])),
                        'ttol_accuracy': float(np.mean([r['correct_prediction'] for r in ttol_results])),
                        'avg_confidence': float(np.mean([r['confidence'] for r in ttol_results])),
                        'results': ttol_results
                    },
                    'random_baseline': random_baseline_stats,
                    'delta_analysis': delta_stats
                }
                
                # Store in probe type results
                probe_type_layer_results[int(layer_num)] = ttol_summary
                
                # Save individual layer file
                ttol_file = layer_dir / 'ttol_analysis.json'
                with open(ttol_file, 'w') as f:
                    json.dump(ttol_summary, f, indent=2)
                
                probe_acc = ttol_summary['probe_performance']['ttol_accuracy']
                probe_loss = ttol_summary['probe_performance']['avg_ttol_loss']
                
                if random_baseline_stats and delta_stats:
                    random_acc = random_baseline_stats['ttol_accuracy']
                    random_loss = random_baseline_stats['avg_ttol_loss']
                    print(f"    Saved 2T1L analysis: {len(ttol_results)} examples")
                    print(f"      Probe:  accuracy: {probe_acc:.3f}, avg_loss: {probe_loss:.3f}")
                    print(f"      Random: accuracy: {random_acc:.3f}, avg_loss: {random_loss:.3f}")
                    print(f"      Delta:  lie_stmt: {delta_stats['avg_lie_statement_delta']:.3f}, truth_stmts: {delta_stats['avg_truth_statements_delta']:.3f}")
                    print(f"      Improvement: {probe_acc - random_acc:.3f} accuracy, {random_loss - probe_loss:.3f} loss")
                else:
                    print(f"    Saved 2T1L analysis: {len(ttol_results)} examples, accuracy: {probe_acc:.3f}, avg_loss: {probe_loss:.3f}")
            else:
                print(f"    No 2T1L results generated for {probe_type} layer {layer_num}")
        
        # After processing all layers for this probe type, create aggregated results
        if probe_type_layer_results:
            layer_data = list(probe_type_layer_results.values())
            
            # Calculate summary statistics across all layers
            probe_accs = [ld['probe_performance']['ttol_accuracy'] for ld in layer_data]
            probe_losses = [ld['probe_performance']['avg_ttol_loss'] for ld in layer_data]
            
            random_accs = []
            random_losses = []
            delta_separations = []
            
            for ld in layer_data:
                if ld.get('random_baseline'):
                    random_accs.append(ld['random_baseline']['ttol_accuracy'])
                    random_losses.append(ld['random_baseline']['avg_ttol_loss'])
                if ld.get('delta_analysis'):
                    delta_separations.append(ld['delta_analysis']['avg_delta_separation'])
            
            # Find best performing layers
            best_acc_layer = max(layer_data, key=lambda x: x['probe_performance']['ttol_accuracy'])
            best_loss_layer = min(layer_data, key=lambda x: x['probe_performance']['avg_ttol_loss'])
            best_delta_layer = None
            if delta_separations:
                best_delta_layer = max(layer_data, key=lambda x: x.get('delta_analysis', {}).get('avg_delta_separation', -999))
            
            probe_type_summary = {
                'probe_type': probe_type,
                'num_layers': len(layer_data),
                'layers': probe_type_layer_results,
                'summary_stats': {
                    'best_layer_by_accuracy': best_acc_layer['layer'],
                    'best_accuracy': best_acc_layer['probe_performance']['ttol_accuracy'],
                    'best_layer_by_loss': best_loss_layer['layer'],
                    'best_loss': best_loss_layer['probe_performance']['avg_ttol_loss'],
                    'best_layer_by_delta': best_delta_layer['layer'] if best_delta_layer else None,
                    'best_delta_separation': best_delta_layer.get('delta_analysis', {}).get('avg_delta_separation') if best_delta_layer else None,
                    'avg_probe_accuracy': float(np.mean(probe_accs)),
                    'avg_probe_loss': float(np.mean(probe_losses)),
                    'avg_random_accuracy': float(np.mean(random_accs)) if random_accs else None,
                    'avg_random_loss': float(np.mean(random_losses)) if random_losses else None,
                    'avg_delta_separation': float(np.mean(delta_separations)) if delta_separations else None,
                    'accuracy_improvement': float(np.mean(probe_accs) - np.mean(random_accs)) if random_accs else None,
                    'loss_improvement': float(np.mean(random_losses) - np.mean(probe_losses)) if random_losses else None
                }
            }
            
            # Store for cross-probe-type comparison
            all_probe_type_summaries[probe_type] = probe_type_summary
            
            # Save aggregated results
            with open(probe_type_dir / 'all_layers_2t1l.json', 'w') as f:
                json.dump(probe_type_summary, f, indent=2)
            
            # Create plots
            create_2t1l_plots(probe_type_dir, probe_type)
            
            print(f"\n  ✓ Saved aggregated 2T1L results for {probe_type}")
            print(f"    📊 Best accuracy: Layer {probe_type_summary['summary_stats']['best_layer_by_accuracy']} ({probe_type_summary['summary_stats']['best_accuracy']:.3f})")
            print(f"    📈 Best loss: Layer {probe_type_summary['summary_stats']['best_layer_by_loss']} ({probe_type_summary['summary_stats']['best_loss']:.3f})")
            if probe_type_summary['summary_stats']['best_layer_by_delta']:
                print(f"    🎯 Best delta: Layer {probe_type_summary['summary_stats']['best_layer_by_delta']} ({probe_type_summary['summary_stats']['best_delta_separation']:.3f})")
            print(f"    🚀 Avg improvement: {probe_type_summary['summary_stats']['accuracy_improvement']:.3f} accuracy")
    
    # Create cross-probe-type comparison
    if all_probe_type_summaries:
        print(f"\n{'='*60}")
        print("🏆 CROSS-PROBE-TYPE PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        # Sort by best accuracy
        probe_types_by_acc = sorted(all_probe_type_summaries.items(), 
                                  key=lambda x: x[1]['summary_stats']['best_accuracy'], 
                                  reverse=True)
        
        print("\n🎯 BEST ACCURACY RANKING:")
        for i, (probe_type, summary) in enumerate(probe_types_by_acc, 1):
            stats = summary['summary_stats']
            print(f"  {i}. {probe_type.upper()}: {stats['best_accuracy']:.3f} (Layer {stats['best_layer_by_accuracy']})")
        
        # Sort by best loss (lower is better)
        probe_types_by_loss = sorted(all_probe_type_summaries.items(), 
                                   key=lambda x: x[1]['summary_stats']['best_loss'])
        
        print("\n📈 BEST LOSS RANKING (lower is better):")
        for i, (probe_type, summary) in enumerate(probe_types_by_loss, 1):
            stats = summary['summary_stats']
            print(f"  {i}. {probe_type.upper()}: {stats['best_loss']:.3f} (Layer {stats['best_layer_by_loss']})")
        
        # Sort by best delta separation
        probe_types_with_delta = [(k, v) for k, v in all_probe_type_summaries.items() 
                                 if v['summary_stats']['best_delta_separation'] is not None]
        if probe_types_with_delta:
            probe_types_by_delta = sorted(probe_types_with_delta, 
                                        key=lambda x: x[1]['summary_stats']['best_delta_separation'], 
                                        reverse=True)
            
            print("\n🎯 BEST DELTA SEPARATION RANKING:")
            for i, (probe_type, summary) in enumerate(probe_types_by_delta, 1):
                stats = summary['summary_stats']
                print(f"  {i}. {probe_type.upper()}: {stats['best_delta_separation']:.3f} (Layer {stats['best_layer_by_delta']})")
        
        # Overall improvement ranking
        probe_types_by_improvement = sorted(all_probe_type_summaries.items(), 
                                          key=lambda x: x[1]['summary_stats']['accuracy_improvement'] or 0, 
                                          reverse=True)
        
        print("\n🚀 OVERALL IMPROVEMENT RANKING:")
        for i, (probe_type, summary) in enumerate(probe_types_by_improvement, 1):
            stats = summary['summary_stats']
            acc_imp = stats['accuracy_improvement'] or 0
            loss_imp = stats['loss_improvement'] or 0
            print(f"  {i}. {probe_type.upper()}: +{acc_imp:.3f} acc, +{loss_imp:.3f} loss")
        
        # Save cross-probe comparison
        cross_comparison = {
            'summary': {
                'total_probe_types': len(all_probe_type_summaries),
                'best_overall_accuracy': probe_types_by_acc[0][1]['summary_stats']['best_accuracy'],
                'best_overall_accuracy_probe': probe_types_by_acc[0][0],
                'best_overall_loss': probe_types_by_loss[0][1]['summary_stats']['best_loss'],
                'best_overall_loss_probe': probe_types_by_loss[0][0],
            },
            'probe_type_summaries': all_probe_type_summaries,
            'rankings': {
                'by_accuracy': [(pt, s['summary_stats']['best_accuracy']) for pt, s in probe_types_by_acc],
                'by_loss': [(pt, s['summary_stats']['best_loss']) for pt, s in probe_types_by_loss],
                'by_delta': [(pt, s['summary_stats']['best_delta_separation']) for pt, s in probe_types_by_delta] if probe_types_with_delta else [],
                'by_improvement': [(pt, s['summary_stats']['accuracy_improvement']) for pt, s in probe_types_by_improvement]
            }
        }
        
        with open(results_path / 'cross_probe_2t1l_comparison.json', 'w') as f:
            json.dump(cross_comparison, f, indent=2)
        
        print(f"\n💾 Saved cross-probe comparison to cross_probe_2t1l_comparison.json")
        print(f"{'='*60}")
    
    print("\n✅ 2T1L analysis complete!")

def main():
    parser = argparse.ArgumentParser(description='Add 2T1L analysis to existing probe results')
    parser.add_argument('--results_dir', type=str, required=True, 
                       help='Directory containing probe evaluation results')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the original 2T1L dataset JSON file')
    
    args = parser.parse_args()
    
    add_2t1l_analysis_to_saved_results(args.results_dir, args.dataset_path)
    print("\n2T1L analysis complete!")

if __name__ == "__main__":
    main()