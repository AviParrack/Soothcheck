import numpy as np
from pathlib import Path
from typing import List, Dict, Union
import json
from jinja2 import Template

def normalize_scores_global(all_scores: List[List[float]]) -> List[List[float]]:
    """Normalize all scores globally to [0, 1] range."""
    # Flatten all scores to find global min/max
    flat_scores = np.array([score for scores in all_scores for score in scores])
    
    if len(flat_scores) == 0:
        return all_scores
    
    min_score = flat_scores.min()
    max_score = flat_scores.max()
    
    if max_score == min_score:
        return [[0.5] * len(scores) for scores in all_scores]  # Set to neutral value
    
    # Normalize to [0, 1] range
    normalized_all = []
    for scores in all_scores:
        scores_array = np.array(scores)
        # Scale to [0, 1] based on global min/max
        normalized = (scores_array - min_score) / (max_score - min_score)
        normalized_all.append(normalized.tolist())
    
    return normalized_all

def score_to_color(score: float) -> str:
    """Convert score (0 to 1) to RGB color string.
    0 = full blue (truth)
    0.5 = transparent (neutral)
    1 = full red (lie)
    """
    # Clamp score to [0, 1] range just in case
    score = max(0, min(1, score))
    
    if score >= 0.5:
        # Red scale for lie predictions (0.5 to 1.0)
        intensity = (score - 0.5) * 2  # Map [0.5, 1] to [0, 1]
        return f"rgba(255, 0, 0, {intensity:.3f})"
    else:
        # Blue scale for truth predictions (0.0 to 0.5)
        intensity = (0.5 - score) * 2  # Map [0, 0.5] to [1, 0]
        return f"rgba(0, 0, 255, {intensity:.3f})"

def clean_llama_tokens(tokens: List[str]) -> List[str]:
    """Clean LLaMA tokenizer artifacts for better HTML display."""
    return [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in tokens]

def generate_token_visualization(token_details: List[Dict], output_path: Path) -> None:
    """Generate HTML visualization of token-level scores."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Token-Level Probe Activations</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .legend {
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 20px 0;
                gap: 20px;
            }
            .legend-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .legend-color {
                width: 20px;
                height: 20px;
                border-radius: 3px;
            }
            .token-container {
                margin: 20px 0;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .token-text {
                line-height: 2;
                margin-bottom: 15px;
                font-size: 16px;
            }
            .token {
                display: inline-block;
                padding: 3px 6px;
                margin: 1px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                border: 1px solid #ddd;
                cursor: help;
                white-space: pre-wrap;
            }
            .metadata {
                font-size: 14px;
                color: #666;
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #eee;
            }
            .metadata-item {
                margin: 5px 0;
            }
            .label-positive {
                color: #d73027;
                font-weight: bold;
            }
            .label-negative {
                color: #1a9850;
                font-weight: bold;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
                margin-top: 10px;
            }
            .stats-item {
                background-color: #f8f9fa;
                padding: 8px;
                border-radius: 5px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Token-Level Probe Activations</h1>
            <p>Visualization of token-level scores from probe analysis</p>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: rgba(0, 0, 255, 1);"></div>
                    <span>Truth Prediction (0)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: rgba(128, 128, 128, 0.2);"></div>
                    <span>Neutral (0.5)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: rgba(255, 0, 0, 1);"></div>
                    <span>Lie Prediction (1)</span>
                </div>
            </div>
        </div>
        
        {% for example in examples %}
        <div class="token-container">
            <div class="token-text">
                {% for token, orig_score, norm_score, color in example.tokens_with_colors %}
                <span class="token" style="background-color: {{ color }}" 
                      title="Original Score: {{ "%.4f"|format(orig_score) }}&#10;Normalized Score: {{ "%.4f"|format(norm_score) }}">{{ token }}</span>
                {% endfor %}
            </div>
            <div class="metadata">
                <div class="metadata-item">
                    <strong>Label:</strong> 
                    <span class="{% if example.label == 1 %}label-positive{% else %}label-negative{% endif %}">
                        {{ example.label }} ({{ "Positive" if example.label == 1 else "Negative" }})
                    </span>
                </div>
                <div class="stats">
                    <div class="stats-item">
                        <strong>Mean Score</strong><br>
                        Original: {{ "%.4f"|format(example.mean_score) }}<br>
                        Normalized: {{ "%.4f"|format(example.norm_mean_score) }}
                    </div>
                    <div class="stats-item">
                        <strong>Score Range</strong><br>
                        Original: {{ "%.4f"|format(example.min_score) }} to {{ "%.4f"|format(example.max_score) }}<br>
                        Normalized: {{ "%.4f"|format(example.norm_min_score) }} to {{ "%.4f"|format(example.norm_max_score) }}
                    </div>
                    <div class="stats-item">
                        <strong>Token Count</strong><br>
                        {{ example.token_count }}
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
        
        <div class="header" style="margin-top: 40px;">
            <p><em>Scores are normalized globally across all examples to [0, 1] range for consistent visualization.<br>
            0 = Truth prediction, 0.5 = Neutral, 1 = Lie prediction</em></p>
        </div>
    </body>
    </html>
    """
    
    # First, collect all scores for global normalization
    all_scores = [detail['token_scores'] for detail in token_details]
    normalized_scores_list = normalize_scores_global(all_scores)
    
    # Prepare data for template
    examples = []
    for i, detail in enumerate(token_details):
        # Clean tokens for display
        clean_tokens = clean_llama_tokens(detail['tokens'])
        normalized_scores = normalized_scores_list[i]
        
        # Create token-color pairs
        tokens_with_colors = [
            (token, orig_score, norm_score, score_to_color(norm_score))
            for token, orig_score, norm_score in zip(
                clean_tokens, 
                detail['token_scores'], 
                normalized_scores
            )
        ]
        
        examples.append({
            'tokens_with_colors': tokens_with_colors,
            'label': detail['label'],
            'mean_score': detail['mean_score'],
            'min_score': detail['min_score'],
            'max_score': detail['max_score'],
            'norm_mean_score': np.mean(normalized_scores),
            'norm_min_score': np.min(normalized_scores),
            'norm_max_score': np.max(normalized_scores),
            'token_count': len(clean_tokens)
        })
    
    # Render template
    template = Template(html_template)
    html_content = template.render(examples=examples)
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML visualization saved to: {output_path}")

