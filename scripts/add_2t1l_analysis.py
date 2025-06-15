import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional

def extract_three_statements_from_tokens(tokens: List[str], token_scores: List[float], 
                                        text: str, lie_statement: str) -> Optional[Dict]:
    """Extract three statements using token-level data directly"""
    
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
        
        # Convert tokens back to readable text for sentence splitting
        response_text = ""
        for token in response_tokens:
            if token.startswith("Ġ"):  # Llama space prefix
                response_text += " " + token[1:]
            elif token in ["\u010a", "Ċ"]:  # Newline tokens
                response_text += "\n"
            else:
                response_text += token
        
        response_text = response_text.strip()
        
        # Simple sentence splitting by periods
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
        
        # Take first 3 sentences
        three_sentences = sentences[:3]
        
        # Map sentences to token ranges (approximate)
        statements = []
        tokens_per_sentence = len(response_tokens) // 3  # Rough estimate
        
        for i, sentence in enumerate(three_sentences):
            start_idx = i * tokens_per_sentence
            end_idx = min((i + 1) * tokens_per_sentence, len(response_tokens)) if i < 2 else len(response_tokens)
            
            statements.append({
                'text': sentence,
                'tokens': response_tokens[start_idx:end_idx],
                'scores': response_scores[start_idx:end_idx],
                'start_idx': assistant_start_idx + start_idx,
                'end_idx': assistant_start_idx + end_idx - 1
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
        statement_mean_scores.append(mean_score)
        statement_texts.append(stmt['text'])
        
        # Convert to logits based on probe type
        if probe_type.lower() == 'logistic':
            # Scores are probabilities [0,1], convert to logits
            p = max(min(mean_score, 0.999), 0.001)  # Clamp to avoid log(0)
            logit = np.log(p / (1 - p))
        else:
            # For other probes, use raw scores as logits
            logit = mean_score
        
        statement_logits.append(logit)
    
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
        'statement_probs': probs.tolist(),
        'lie_statement_idx': lie_claim_idx,
        'predicted_lie_idx': predicted_lie_idx,
        'ttol_loss': normalized_loss.item(),
        'correct_prediction': predicted_lie_idx == lie_claim_idx,
        'confidence': probs[predicted_lie_idx].item(),
        'cross_entropy_loss': cross_entropy_loss.item()
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
        'random_logits': random_logits,
        'statement_probs': probs.tolist(),
        'lie_statement_idx': lie_claim_idx,
        'predicted_lie_idx': predicted_lie_idx,
        'ttol_loss': normalized_loss.item(),
        'correct_prediction': predicted_lie_idx == lie_claim_idx,
        'confidence': probs[predicted_lie_idx].item(),
        'cross_entropy_loss': cross_entropy_loss.item()
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
            'statement_idx': i,
            'is_lie_statement': i == lie_statement_idx,
            'truth_mean_score': float(truth_mean),
            'lie_mean_score': float(lie_mean),
            'delta': float(delta),
            'statement_text': truth_stmt['text']
        })
    
    # Calculate summary statistics
    lie_stmt_delta = statement_deltas[lie_statement_idx]['delta']
    truth_stmt_deltas = [s['delta'] for s in statement_deltas if not s['is_lie_statement']]
    
    return {
        'lie_statement_idx': lie_statement_idx,
        'lie_statement_delta': lie_stmt_delta,
        'truth_statements_deltas': truth_stmt_deltas,
        'avg_truth_statements_delta': float(np.mean(truth_stmt_deltas)),
        'statement_deltas': statement_deltas,
        'delta_separation': lie_stmt_delta - np.mean(truth_stmt_deltas),  # How much higher lie statement delta is
        'good_separation': lie_stmt_delta > np.mean(truth_stmt_deltas)  # True if lie statement has higher delta
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
    
    # Process each probe type
    for probe_type_dir in results_path.iterdir():
        if not probe_type_dir.is_dir() or probe_type_dir.name.startswith('.'):
            continue
        
        probe_type = probe_type_dir.name
        print(f"\nProcessing {probe_type} probes for 2T1L analysis")
        
        # Process each layer
        for layer_dir in probe_type_dir.iterdir():
            if not layer_dir.is_dir() or not layer_dir.name.startswith('layer_'):
                continue
            
            layer_num = layer_dir.name.split('_')[1]
            sample_scores_file = layer_dir / 'sample_token_scores.json'
            
            if not sample_scores_file.exists():
                continue
            
            print(f"  Processing {probe_type} layer {layer_num}")
            
            # Load the saved sample scores
            with open(sample_scores_file, 'r') as f:
                samples = json.load(f)
            
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