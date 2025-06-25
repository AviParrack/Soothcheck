"""
NTML Dataset Generation Scripts

This module contains scripts for generating conversational NTML (N Truths, M Lies) datasets
from statement banks. The generated datasets are formatted for training statement-level 
truth detection probes.

Main components:
- config.py: Configuration for target ratios and generation parameters
- statement_loader.py: Loading and managing statement banks
- ntml_generator.py: Core generation logic with sampling and shuffling
- formatters.py: Conversation formatting utilities
- generate_datasets.py: Main script to generate all datasets
"""

__version__ = "0.1.0" 