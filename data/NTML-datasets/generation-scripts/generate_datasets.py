#!/usr/bin/env python3
"""
Main script for generating NTML datasets.

This script generates conversational truth/lie datasets for all configured ratios
and saves them as JSONL files for statement-level probing experiments.

Usage:
    python generate_datasets.py [--samples N] [--seed N] [--output-dir PATH] [--ratios RATIO1,RATIO2,...]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add probity to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import GenerationConfig, DEFAULT_RATIOS, DEFAULT_CONFIG
from ntml_generator import DatasetGenerator

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ntml_generation.log')
        ]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate NTML (N Truths, M Lies) conversational datasets"
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=500,
        help='Number of samples to generate per ratio (default: 500)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible generation (default: 42)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for generated datasets (default: auto-detected probity/data/NTML-datasets)'
    )
    
    parser.add_argument(
        '--ratios',
        type=str,
        default=None,
        help='Comma-separated list of ratios to generate (e.g., "2T1L,5T1L"). If not specified, generates all default ratios.'
    )
    
    parser.add_argument(
        '--truth-bank',
        type=str,
        default=None,
        help='Path to truth statement bank CSV file (default: auto-detected probity/data/statement-banks/truth_bank.csv)'
    )
    
    parser.add_argument(
        '--lie-bank',
        type=str,
        default=None,
        help='Path to lie statement bank CSV file (default: auto-detected probity/data/statement-banks/lie_bank.csv)'
    )
    
    parser.add_argument(
        '--no-timestamps',
        action='store_true',
        help='Disable timestamps in generated conversations'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable output validation during generation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def parse_ratios(ratio_string: str) -> List[dict]:
    """Parse ratio string into ratio configuration list.
    
    Args:
        ratio_string: Comma-separated ratio names (e.g., "2T1L,5T1L")
        
    Returns:
        List of ratio configuration dictionaries
    """
    if not ratio_string:
        return DEFAULT_RATIOS
    
    ratio_names = [name.strip() for name in ratio_string.split(',')]
    ratio_configs = []
    
    # Create ratio name to config mapping
    ratio_map = {ratio['name']: ratio for ratio in DEFAULT_RATIOS}
    
    for ratio_name in ratio_names:
        if ratio_name in ratio_map:
            ratio_configs.append(ratio_map[ratio_name])
        else:
            # Try to parse custom ratio format (e.g., "3T2L")
            try:
                if 'T' in ratio_name and 'L' in ratio_name:
                    parts = ratio_name.split('T')
                    truths = int(parts[0])
                    lies = int(parts[1].replace('L', ''))
                    
                    ratio_configs.append({
                        'name': ratio_name,
                        'truths': truths,
                        'lies': lies
                    })
                else:
                    raise ValueError(f"Invalid ratio format: {ratio_name}")
            except (ValueError, IndexError) as e:
                logging.error(f"Could not parse ratio '{ratio_name}': {e}")
                sys.exit(1)
    
    return ratio_configs

def validate_config(config: GenerationConfig) -> bool:
    """Validate generation configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check if statement bank files exist
    truth_path = Path(config.truth_bank_path)
    if not truth_path.exists():
        logging.error(f"Truth bank file not found: {truth_path}")
        return False
    
    lie_path = Path(config.lie_bank_path)
    if not lie_path.exists():
        logging.error(f"Lie bank file not found: {lie_path}")
        return False
    
    # Validate ratios
    if not config.target_ratios:
        logging.error("No target ratios specified")
        return False
    
    for ratio in config.target_ratios:
        if ratio['truths'] < 1 or ratio['lies'] < 1:
            logging.error(f"Invalid ratio {ratio['name']}: truths and lies must be >= 1")
            return False
    
    return True

def main():
    """Main generation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting NTML dataset generation")
    logger.info(f"Command line args: {vars(args)}")
    
    try:
        # Parse ratios
        target_ratios = parse_ratios(args.ratios)
        logger.info(f"Target ratios: {[r['name'] for r in target_ratios]}")
        
        # Create configuration
        config = GenerationConfig(
            samples_per_ratio=args.samples,
            random_seed=args.seed,
            truth_bank_path=args.truth_bank or DEFAULT_CONFIG.truth_bank_path,
            lie_bank_path=args.lie_bank or DEFAULT_CONFIG.lie_bank_path,
            output_dir=args.output_dir or DEFAULT_CONFIG.output_dir,
            target_ratios=target_ratios,
            add_timestamps=not args.no_timestamps,
            validate_output=not args.no_validation
        )
        
        # Validate configuration
        if not validate_config(config):
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        # Generate datasets
        generator = DatasetGenerator(config)
        datasets = generator.generate_and_save_all()
        
        # Log summary statistics
        logger.info("=" * 50)
        logger.info("GENERATION SUMMARY")
        logger.info("=" * 50)
        
        total_conversations = 0
        for ratio_name, conversations in datasets.items():
            count = len(conversations)
            total_conversations += count
            output_file = config.get_output_filename(ratio_name)
            logger.info(f"{ratio_name}: {count} conversations → {output_file}")
        
        logger.info(f"Total conversations generated: {total_conversations}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info("Generation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 