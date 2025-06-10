import numpy as np
from pathlib import Path
from typing import List, Dict, Union
import json
from jinja2 import Template

def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range."""
    scores = np.array(scores)
    min_score = scores.min()
    max_score = scores.max()
    if max_score == min_score:
        return [0.5] * len(scores)
    return ((scores - min_score) / (max_score - min_score)).tolist()

def score_to_color(score: float, is_positive: bool = True) -> str:
    """Convert score to RGB color string."""
    if is_positive:
        # Red scale for positive class (score -> red intensity)
        return f"rgba(255, 0, 0, {score:.3f})"
    else:
        # Blue scale for negative class (score -> blue intensity)
        return f"rgba(0, 0, 255, {score:.3f})"

def generate_token_visualization(token_details: List[Dict], output_path: Path) -> None:
    """Generate HTML visualization of token-level scores."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .token-container {
                margin: 20px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            .token {
                display: inline-block;
                padding: 2px 4px;
                margin: 0 2px;
                border-radius: 3px;
                font-family: monospace;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Token-Level Probe Activations</h1>
        {% for example in examples %}
        <div class="token-container">
            <div>
                {% for token, score, color in example.tokens_with_colors %}
                <span class="token" style="background-color: {{ color }}" title="Score: {{ score }}">
                    {{ token }}
                </span>
                {% endfor %}
            </div>
            <div class="metadata">
                <p>Label: {{ example.label }} ({{ "Positive" if example.label == 1 else "Negative" }})</p>
                <p>Mean Score: {{ "%.3f"|format(example.mean_score) }}</p>
                <p>Range: {{ "%.3f"|format(example.min_score) }} - {{ "%.3f"|format(example.max_score) }}</p>
            </div>
        </div>
        {% endfor %}
    </body>
    </html>
    """
    
    # Prepare data for template
    examples = []
    for detail in token_details:
        # Normalize scores for better visualization
        normalized_scores = normalize_scores(detail['token_scores'])
        
        # Create token-color pairs
        tokens_with_colors = [
            (token, score, score_to_color(norm_score, detail['label'] == 1))
            for token, score, norm_score in zip(
                detail['tokens'], 
                detail['token_scores'], 
                normalized_scores
            )
        ]
        
        examples.append({
            'tokens_with_colors': tokens_with_colors,
            'label': detail['label'],
            'mean_score': detail['mean_score'],
            'min_score': detail['min_score'],
            'max_score': detail['max_score']
        })
    
    # Render template
    template = Template(html_template)
    html_content = template.render(examples=examples)
    
    # Save HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

def save_token_scores_csv(token_details: List[Dict], output_path: Path) -> None:
    """Save token scores in CSV format for additional analysis."""
    import pandas as pd
    
    # Prepare data for DataFrame
    rows = []
    for detail in token_details:
        for token, score in zip(detail['tokens'], detail['token_scores']):
            rows.append({
                'text': detail['text'],
                'label': detail['label'],
                'token': token,
                'score': score
            })
    
    # Create and save DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)