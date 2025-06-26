"""
Statement bank loading and management utilities.

Loads truth and lie statements from CSV files and provides sampling functionality
with deduplication and validation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StatementBankLoader:
    """Loads and manages truth/lie statement banks."""
    
    def __init__(self, truth_path: str, lie_path: str):
        """Initialize loader with paths to truth and lie bank CSV files.
        
        Args:
            truth_path: Path to truth_bank.csv
            lie_path: Path to lie_bank.csv
        """
        self.truth_path = Path(truth_path)
        self.lie_path = Path(lie_path)
        self.truth_bank = None
        self.lie_bank = None
        self._loaded = False
    
    def load_banks(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both truth and lie statement banks.
        
        Returns:
            Tuple of (truth_bank_df, lie_bank_df)
        """
        logger.info(f"Loading truth bank from {self.truth_path}")
        self.truth_bank = pd.read_csv(self.truth_path)
        
        logger.info(f"Loading lie bank from {self.lie_path}")
        self.lie_bank = pd.read_csv(self.lie_path)
        
        self._validate_banks()
        self._loaded = True
        
        logger.info(f"Loaded {len(self.truth_bank)} truth statements and {len(self.lie_bank)} lie statements")
        
        return self.truth_bank, self.lie_bank
    
    def _validate_banks(self):
        """Validate loaded statement banks."""
        required_columns = ['statement', 'label', 'statement_id', 'source']
        
        # Check truth bank
        missing_truth_cols = set(required_columns) - set(self.truth_bank.columns)
        if missing_truth_cols:
            raise ValueError(f"Truth bank missing columns: {missing_truth_cols}")
        
        # Check lie bank
        missing_lie_cols = set(required_columns) - set(self.lie_bank.columns)
        if missing_lie_cols:
            raise ValueError(f"Lie bank missing columns: {missing_lie_cols}")
        
        # Validate labels
        truth_labels = self.truth_bank['label'].unique()
        if not all(label == 1 for label in truth_labels):
            logger.warning(f"Truth bank contains non-1 labels: {truth_labels}")
        
        lie_labels = self.lie_bank['label'].unique()
        if not all(label == 0 for label in lie_labels):
            logger.warning(f"Lie bank contains non-0 labels: {lie_labels}")
    
    def get_bank_statistics(self) -> Dict[str, any]:
        """Get statistics about the loaded banks."""
        if not self._loaded:
            raise RuntimeError("Banks not loaded. Call load_banks() first.")
        
        return {
            "truth_count": len(self.truth_bank),
            "lie_count": len(self.lie_bank),
            "truth_sources": self.truth_bank['source'].value_counts().to_dict(),
            "lie_sources": self.lie_bank['source'].value_counts().to_dict(),
            "total_sources": len(set(self.truth_bank['source']) | set(self.lie_bank['source']))
        }
    
    def sample_statements(self, n_truths: int, n_lies: int, random_state: int = None) -> Tuple[List[Dict], List[Dict]]:
        """Sample statements from the banks.
        
        Args:
            n_truths: Number of truth statements to sample
            n_lies: Number of lie statements to sample
            random_state: Random seed for reproducible sampling
            
        Returns:
            Tuple of (sampled_truths, sampled_lies) as list of dicts
        """
        if not self._loaded:
            raise RuntimeError("Banks not loaded. Call load_banks() first.")
        
        if n_truths > len(self.truth_bank):
            raise ValueError(f"Requested {n_truths} truths but only {len(self.truth_bank)} available")
        
        if n_lies > len(self.lie_bank):
            raise ValueError(f"Requested {n_lies} lies but only {len(self.lie_bank)} available")
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample truth statements
        truth_sample = self.truth_bank.sample(n=n_truths, random_state=random_state)
        sampled_truths = truth_sample.to_dict('records')
        
        # Sample lie statements
        lie_sample = self.lie_bank.sample(n=n_lies, random_state=random_state)
        sampled_lies = lie_sample.to_dict('records')
        
        logger.debug(f"Sampled {len(sampled_truths)} truths and {len(sampled_lies)} lies")
        
        return sampled_truths, sampled_lies
    
    def check_statement_overlap(self, statements: List[str]) -> Set[str]:
        """Check if any statements appear in both banks (potential duplicates).
        
        Args:
            statements: List of statement texts to check
            
        Returns:
            Set of overlapping statements
        """
        if not self._loaded:
            raise RuntimeError("Banks not loaded. Call load_banks() first.")
        
        statement_set = set(statements)
        truth_statements = set(self.truth_bank['statement'])
        lie_statements = set(self.lie_bank['statement'])
        
        # Find statements that appear in both banks
        bank_overlap = truth_statements & lie_statements
        
        # Find input statements that overlap with banks
        input_overlap = statement_set & (truth_statements | lie_statements)
        
        if bank_overlap:
            logger.warning(f"Found {len(bank_overlap)} statements in both truth and lie banks")
        
        return input_overlap
    
    def get_source_distribution(self, statements: List[Dict]) -> Dict[str, int]:
        """Get distribution of statement sources.
        
        Args:
            statements: List of statement dictionaries with 'source' key
            
        Returns:
            Dictionary mapping source names to counts
        """
        sources = [stmt['source'] for stmt in statements if 'source' in stmt]
        return pd.Series(sources).value_counts().to_dict()


def load_statement_banks(truth_path: str, lie_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to load statement banks.
    
    Args:
        truth_path: Path to truth_bank.csv
        lie_path: Path to lie_bank.csv
        
    Returns:
        Tuple of (truth_bank_df, lie_bank_df)
    """
    loader = StatementBankLoader(truth_path, lie_path)
    return loader.load_banks() 