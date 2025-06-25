#!/usr/bin/env python3
"""
Script to process statement bank CSV files.

This script:
1. Goes through every CSV file in the mixed-statements directory
2. Removes duplicates from each file
3. Balances the labels (50/50 split of true and false)
4. Truncates to max 1300 entries while preserving balance
5. Combines all processed files into truth_bank.csv and lie_bank.csv
"""

import pandas as pd
import os
from pathlib import Path
import logging
from typing import Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_csv(file_path: Path, source_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a single CSV file according to the requirements.
    
    Args:
        file_path: Path to the CSV file
        source_name: Name to use as source identifier
        
    Returns:
        Tuple of (true_statements_df, false_statements_df)
    """
    logger.info(f"Processing {file_path.name}...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    logger.info(f"  Initial rows: {len(df)}")
    
    # Remove duplicates based on statement text
    df_deduplicated = df.drop_duplicates(subset=['statement'], keep='first')
    duplicates_removed = len(df) - len(df_deduplicated)
    if duplicates_removed > 0:
        logger.info(f"  Removed {duplicates_removed} duplicates")
    
    # Separate true and false statements
    true_statements = df_deduplicated[df_deduplicated['label'] == 1].copy()
    false_statements = df_deduplicated[df_deduplicated['label'] == 0].copy()
    
    logger.info(f"  True statements: {len(true_statements)}, False statements: {len(false_statements)}")
    
    # Balance the labels (take minimum count for 50/50 split)
    min_count = min(len(true_statements), len(false_statements))
    
    # If we have more than 650 of each (since max total is 1300), limit to 650 each
    max_per_label = min(min_count, 650)
    
    true_statements = true_statements.sample(n=max_per_label, random_state=42).reset_index(drop=True)
    false_statements = false_statements.sample(n=max_per_label, random_state=42).reset_index(drop=True)
    
    logger.info(f"  Balanced to {len(true_statements)} true and {len(false_statements)} false statements")
    
    # Add source column
    true_statements['source'] = source_name
    false_statements['source'] = source_name
    
    return true_statements, false_statements

def create_final_banks(all_true_statements: List[pd.DataFrame], all_false_statements: List[pd.DataFrame], 
                      output_dir: Path) -> None:
    """
    Combine all processed statements into final bank files.
    
    Args:
        all_true_statements: List of DataFrames containing true statements
        all_false_statements: List of DataFrames containing false statements
        output_dir: Directory to save the output files
    """
    logger.info("Creating final bank files...")
    
    # Combine all true statements
    combined_true = pd.concat(all_true_statements, ignore_index=True)
    combined_true['statement_id'] = range(1, len(combined_true) + 1)
    
    # Combine all false statements  
    combined_false = pd.concat(all_false_statements, ignore_index=True)
    combined_false['statement_id'] = range(1, len(combined_false) + 1)
    
    # Reorder columns to match specification
    column_order = ['statement', 'label', 'statement_id', 'source']
    combined_true = combined_true[column_order]
    combined_false = combined_false[column_order]
    
    # Save to CSV files
    truth_bank_path = output_dir / 'truth_bank.csv'
    lie_bank_path = output_dir / 'lie_bank.csv'
    
    combined_true.to_csv(truth_bank_path, index=False)
    combined_false.to_csv(lie_bank_path, index=False)
    
    logger.info(f"Saved {len(combined_true)} true statements to {truth_bank_path}")
    logger.info(f"Saved {len(combined_false)} false statements to {lie_bank_path}")

def main():
    """Main function to process all CSV files."""
    # Define paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data' / 'statement-banks'
    mixed_statements_dir = data_dir / 'mixed-statements'
    
    # Check if directories exist
    if not mixed_statements_dir.exists():
        logger.error(f"Mixed statements directory not found: {mixed_statements_dir}")
        return
    
    # Get all CSV files
    csv_files = list(mixed_statements_dir.glob('*.csv'))
    if not csv_files:
        logger.error(f"No CSV files found in {mixed_statements_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    all_true_statements = []
    all_false_statements = []
    
    for csv_file in csv_files:
        # Use filename without extension as source name
        source_name = csv_file.stem.replace('_true_false', '')
        
        try:
            true_stmts, false_stmts = process_single_csv(csv_file, source_name)
            all_true_statements.append(true_stmts)
            all_false_statements.append(false_stmts)
        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")
            continue
    
    if not all_true_statements:
        logger.error("No files were successfully processed")
        return
    
    # Create final bank files
    create_final_banks(all_true_statements, all_false_statements, data_dir)
    
    # Print summary statistics
    total_true = sum(len(df) for df in all_true_statements)
    total_false = sum(len(df) for df in all_false_statements)
    logger.info("=" * 50)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Total files processed: {len(all_true_statements)}")
    logger.info(f"Total true statements: {total_true}")
    logger.info(f"Total false statements: {total_false}")
    logger.info(f"Output files saved to: {data_dir}")

if __name__ == "__main__":
    main() 