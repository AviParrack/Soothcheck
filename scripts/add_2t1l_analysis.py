import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional

def extract_three_statements_from_tokens(tokens: List[str], token_scores: List[float], 
                                        text: str, lie_statement: str) -> Optional[Dict]:
    """Extract three statements using token-level data with proper sentence-to-token mapping"""
    
    try:
        assistant_start_idx = None
        assistant_end_idx = None
        
        # Look for the assistant header pattern
        for i in range(len(tokens) - 3):
            if (tokens[i] == "<|start_header_id|>" and 
                tokens[i+1] == "assistant" and 
                tokens[i+2] == "<|end_header_id|>"):
                assistant_start_idx = i + 3
                break
        
        # Find the end token
        if assistant_start_idx:
            for i in range(assistant_start_idx, len(tokens)):
                if tokens[i] == "<|eot_id|>":
                    assistant_end_idx = i
                    break
        
        if assistant_start_idx is None or assistant_end_idx is None:
            return None
        
        # Extract assistant response tokens and scores
        response_tokens = tokens[assistant_start_idx:assistant_end_idx]
        response_scores = token_scores[assistant_start_idx:assistant_end_idx]
        
        # Create character-to-token mapping by reconstructing text step by step
        char_to_token = {}
        current_char_pos = 0
        
        for token_idx, token in enumerate(response_tokens):
            # Convert token to its text representation
            if token.startswith("Ġ"):  # Llama space prefix
                token_text = " " + token[1:]
            elif token in ["\u010a", "Ċ"]:  # Newline tokens
                token_text = "\n"
            else:
                token_text = token
            
            # Map each character position to this token
            for _ in range(len(token_text)):
                char_to_token[current_char_pos] = token_idx
                current_char_pos += 1
        
        # Reconstruct the full response text
        response_text = ""
        for token in response_tokens:
            if token.startswith("Ġ"):  # Llama space prefix
                response_text += " " + token[1:]
            elif token in ["\u010a", "Ċ"]:  # Newline tokens
                response_text += "\n"
            else:
                response_text += token
        
        response_text = response_text.strip()
        
        # Find sentence boundaries by looking for periods followed by space or newline
        sentence_boundaries = []
        for i, char in enumerate(response_text):
            if char == '.' and (i == len(response_text) - 1 or 
                               response_text[i + 1] in [' ', '\n', '\t']):
                sentence_boundaries.append(i + 1)  # Include the period
        
        # If we don't have enough sentence boundaries, split differently
        if len(sentence_boundaries) < 2:
            # Fallback: split by newlines or roughly into thirds
            newline_positions = [i for i, char in enumerate(response_text) if char == '\n']
            if len(newline_positions) >= 2:
                sentence_boundaries = newline_positions[:2] + [len(response_text)]
            else:
                # Last resort: split into thirds
                third = len(response_text) // 3
                sentence_boundaries = [third, 2 * third, len(response_text)]
        
        # Extract sentences based on boundaries
        sentences = []
        sentence_token_ranges = []
        
        start_char = 0
        for boundary in sentence_boundaries[:3]:  # Only take first 3 boundaries
            # Get the sentence text
            sentence_text = response_text[start_char:boundary].strip()
            
            if len(sentence_text.split()) > 2:  # Only substantial sentences
                # Map character positions to token indices
                start_token_idx = char_to_token.get(start_char, 0)
                end_char = min(boundary - 1, len(response_text) - 1)
                end_token_idx = char_to_token.get(end_char, len(response_tokens) - 1)
                
                # Ensure we have a valid range
                if end_token_idx < start_token_idx:
                    end_token_idx = start_token_idx
                
                sentences.append(sentence_text)
                sentence_token_ranges.append((start_token_idx, end_token_idx + 1))  # +1 for exclusive end
            
            start_char = boundary
            
            # If we have 3 sentences, stop
            if len(sentences) >= 3:
                break
        
        # If we still don't have 3 sentences, fall back to equal division
        if len(sentences) < 3:
            print(f"Warning: Only found {len(sentences)} sentences, falling back to equal division")
            
            # Simple sentence splitting by periods (fallback)
            sentences = []
            current_start = 0
            
            for i, char in enumerate(response_text):
                if char == '.' and i < len(response_text) - 1:
                    sentence = response_text[current_start:i+1].strip()
                    if len(sentence.split()) > 3:  # Only substantial sentences
                        sentences.append(sentence)
                    current_start = i + 1
            
            # Add remaining text as last sentence
            remaining = response_text[current_start:].strip()
            if remaining and len(remaining.split()) > 3:
                sentences.append(remaining)
            
            if len(sentences) < 3:
                return None
            
            # Take first 3 sentences and divide tokens equally
            three_sentences = sentences[:3]
            tokens_per_sentence = len(response_tokens) // 3
            
            sentence_token_ranges = []
            for i in range(3):
                start_idx = i * tokens_per_sentence
                end_idx = min((i + 1) * tokens_per_sentence, len(response_tokens)) if i < 2 else len(response_tokens)
                sentence_token_ranges.append((start_idx, end_idx))
            
            sentences = three_sentences
        
        # Build final statements with proper token alignment
        statements = []
        for i, (sentence_text, (start_token_idx, end_token_idx)) in enumerate(zip(sentences[:3], sentence_token_ranges[:3])):
            # Ensure indices are within bounds
            start_token_idx = max(0, min(start_token_idx, len(response_tokens) - 1))
            end_token_idx = max(start_token_idx + 1, min(end_token_idx, len(response_tokens)))
            
            stmt_tokens = response_tokens[start_token_idx:end_token_idx]
            stmt_scores = response_scores[start_token_idx:end_token_idx]
            
            # print(f"Sentence {i}: '{sentence_text}'")
            # print(f"  Tokens ({start_token_idx}-{end_token_idx}): {stmt_tokens}")
            # print(f"  Scores: {[f'{s:.3f}' for s in stmt_scores[:5]]}...")  # Show first 5 scores
            
            statements.append({
                'text': sentence_text,
                'tokens': stmt_tokens,
                'scores': stmt_scores,
                'start_idx': assistant_start_idx + start_token_idx,
                'end_idx': assistant_start_idx + end_token_idx - 1
            })
        
        # Identify which contains the lie
        lie_statement_idx = identify_lie_statement_simple(statements, lie_statement)
        
        return {
            'statements': statements,
            'lie_statement_idx': lie_statement_idx,
            'assistant_start_idx': assistant_start_idx,
            'assistant_end_idx': assistant_end_idx
        }
        
    except Exception as e:
        print(f"Error in extract_three_statements_from_tokens: {e}")
        return None

def identify_lie_statement_simple(statements: List[Dict], lie_statement: str) -> Optional[int]:
    """Simple lie statement identification"""
    lie_words = set(lie_statement.lower().split())
    best_match_idx = None
    best_score = 0
    
    for i, stmt in enumerate(statements):
        stmt_text = stmt['text'].lower()
        stmt_words = set(stmt_text.split())
        
        # Word overlap
        intersection = lie_words.intersection(stmt_words)
        overlap_ratio = len(intersection) / len(lie_words) if lie_words else 0
        
        # Substring matching
        substring_score = 1.0 if lie_statement.lower() in stmt_text else 0
        
        combined_score = max(overlap_ratio, substring_score)
        
        if combined_score > best_score and combined_score > 0.3:
            best_score = combined_score
            best_match_idx = i
    
    return best_match_idx

def analyze_2t1l_from_saved_scores(statements_info: Dict, lie_statement: str, 
                                  probe_type: str) -> Optional[Dict]:
    """Analyze 2T1L loss using saved token scores"""
    
    if statements_info['lie_statement_idx'] is None:
        return None
    
    # Calculate mean score for each statement
    statement_logits = []
    statement_texts = []
    statement_mean_scores = []
    
    for i, stmt in enumerate(statements_info['statements']):
        mean_score = np.mean(stmt['scores'])
        statement_mean_scores.append(float(mean_score))
        statement_texts.append(stmt['text'])
        
        # Convert to logits based on probe type
        if probe_type.lower() == 'logistic':
            # Scores are probabilities [0,1], convert to logits
            p = max(min(mean_score, 0.999), 0.001)  # Clamp to avoid log(0)
            logit = np.log(p / (1 - p))
        else:
            # For other probes, use raw scores as logits
            logit = mean_score
        
        statement_logits.append(float(logit))
    
    # Apply softmax to get probability distribution
    logits_tensor = torch.tensor(statement_logits, dtype=torch.float32)
    probs = torch.softmax(logits_tensor, dim=0)
    
    # Calculate 2T1L loss
    lie_claim_idx = statements_info['lie_statement_idx']
    target = torch.zeros(3)
    target[lie_claim_idx] = 1.0
    
    # Cross-entropy loss
    cross_entropy_loss = -torch.sum(target * torch.log(probs + 1e-8))
    
    # Subtract random baseline (log(3))
    normalized_loss = cross_entropy_loss - np.log(3)
    
    predicted_lie_idx = torch.argmax(probs).item()
    
    return {
        'statements': statement_texts,
        'statement_mean_scores': statement_mean_scores,
        'statement_logits': statement_logits,
        'statement_probs': [float(x) for x in probs.tolist()],
        'lie_statement_idx': int(lie_claim_idx),
        'predicted_lie_idx': int(predicted_lie_idx),
        'ttol_loss': float(normalized_loss.item()),
        'correct_prediction': bool(predicted_lie_idx == lie_claim_idx),
        'confidence': float(probs[predicted_lie_idx].item()),
        'cross_entropy_loss': float(cross_entropy_loss.item())
    }

def analyze_2t1l_random_baseline(statements_info: Dict, lie_statement: str) -> Optional[Dict]:
    """Analyze 2T1L loss using random guessing baseline"""
    
    if statements_info['lie_statement_idx'] is None:
        return None
    
    # Generate random logits (uniform distribution)
    import random
    random_logits = [random.uniform(-1, 1) for _ in range(3)]
    
    # Apply softmax to get probability distribution
    logits_tensor = torch.tensor(random_logits, dtype=torch.float32)
    probs = torch.softmax(logits_tensor, dim=0)
    
    # Calculate 2T1L loss
    lie_claim_idx = statements_info['lie_statement_idx']
    target = torch.zeros(3)
    target[lie_claim_idx] = 1.0
    
    # Cross-entropy loss
    cross_entropy_loss = -torch.sum(target * torch.log(probs + 1e-8))
    
    # Subtract random baseline (log(3))
    normalized_loss = cross_entropy_loss - np.log(3)
    
    predicted_lie_idx = torch.argmax(probs).item()
    
    return {
        'method': 'random_baseline',
        'random_logits': [float(x) for x in random_logits],
        'statement_probs': [float(x) for x in probs.tolist()],
        'lie_statement_idx': int(lie_claim_idx),
        'predicted_lie_idx': int(predicted_lie_idx),
        'ttol_loss': float(normalized_loss.item()),
        'correct_prediction': bool(predicted_lie_idx == lie_claim_idx),
        'confidence': float(probs[predicted_lie_idx].item()),
        'cross_entropy_loss': float(cross_entropy_loss.item())
    }

def extract_assistant_response_from_chat(full_chat_text: str) -> Optional[str]:
    """Extract just the assistant's response from the full chat template text"""
    
    # Look for the assistant response pattern
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
    eot_marker = "<|eot_id|>"
    
    assistant_start = full_chat_text.find(assistant_marker)
    if assistant_start == -1:
        return None
    
    # Find the actual start of the response (after the header)
    response_start = assistant_start + len(assistant_marker)
    
    # Find the end of the response
    eot_pos = full_chat_text.find(eot_marker, response_start)
    if eot_pos == -1:
        response_text = full_chat_text[response_start:].strip()
    else:
        response_text = full_chat_text[response_start:eot_pos].strip()
    
    # Clean up any extra whitespace/newlines
    response_text = response_text.strip()
    
    return response_text if response_text else None

def analyze_statement_deltas(truth_statements_info: Dict, lie_statements_info: Dict, 
                           lie_statement: str) -> Optional[Dict]:
    """Analyze score deltas between truth and lie versions for each statement"""
    
    if (len(truth_statements_info['statements']) != 3 or 
        len(lie_statements_info['statements']) != 3 or
        truth_statements_info['lie_statement_idx'] is None):
        return None
    
    lie_statement_idx = truth_statements_info['lie_statement_idx']
    statement_deltas = []
    
    # Calculate delta for each statement
    for i in range(3):
        truth_stmt = truth_statements_info['statements'][i]
        lie_stmt = lie_statements_info['statements'][i]
        
        # Calculate mean scores for this statement
        truth_mean = np.mean(truth_stmt['scores'])
        lie_mean = np.mean(lie_stmt['scores'])
        
        # Delta = lie_score - truth_score (positive means higher in lie version)
        delta = lie_mean - truth_mean
        
        statement_deltas.append({
            'statement_idx': int(i),
            'is_lie_statement': bool(i == lie_statement_idx),
            'truth_mean_score': float(truth_mean),
            'lie_mean_score': float(lie_mean),
            'delta': float(delta),
            'statement_text': truth_stmt['text']
        })
    
    # Calculate summary statistics
    lie_stmt_delta = statement_deltas[lie_statement_idx]['delta']
    truth_stmt_deltas = [s['delta'] for s in statement_deltas if not s['is_lie_statement']]
    
    return {
        'lie_statement_idx': int(lie_statement_idx),
        'lie_statement_delta': float(lie_stmt_delta),
        'truth_statements_deltas': [float(x) for x in truth_stmt_deltas],
        'avg_truth_statements_delta': float(np.mean(truth_stmt_deltas)),
        'statement_deltas': statement_deltas,
        'delta_separation': float(lie_stmt_delta - np.mean(truth_stmt_deltas)),  # How much higher lie statement delta is
        'good_separation': bool(lie_stmt_delta > np.mean(truth_stmt_deltas))  # True if lie statement has higher delta
    }

def calculate_delta_statistics(ttol_results: List[Dict]) -> Dict:
    """Calculate aggregate delta statistics across all examples"""
    
    delta_analyses = [r['delta_analysis'] for r in ttol_results if 'delta_analysis' in r]
    
    if not delta_analyses:
        return {}
    
    # Aggregate statistics
    lie_deltas = [d['lie_statement_delta'] for d in delta_analyses]
    truth_deltas = []
    for d in delta_analyses:
        truth_deltas.extend(d['truth_statements_deltas'])
    
    separations = [d['delta_separation'] for d in delta_analyses]
    good_separations = [d['good_separation'] for d in delta_analyses]
    
    return {
        'avg_lie_statement_delta': float(np.mean(lie_deltas)),
        'avg_truth_statements_delta': float(np.mean(truth_deltas)),
        'avg_delta_separation': float(np.mean(separations)),
        'separation_accuracy': float(np.mean(good_separations)),  # Fraction where lie > truth deltas
        'std_lie_statement_delta': float(np.std(lie_deltas)),
        'std_truth_statements_delta': float(np.std(truth_deltas)),
        'min_separation': float(np.min(separations)),
        'max_separation': float(np.max(separations)),
        'num_examples': len(delta_analyses)
    }

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