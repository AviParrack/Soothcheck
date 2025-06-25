"""
Core NTML dataset generation logic.

Implements sampling, shuffling, and dataset generation for conversational
truth/lie datasets with statement-level labeling.
"""

import random
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime

from config import GenerationConfig, DEFAULT_CONFIG
from statement_loader import StatementBankLoader
from formatters import ConversationFormatter

logger = logging.getLogger(__name__)

class NTMLSampler:
    """Handles sampling and shuffling of truth/lie statements."""
    
    def __init__(self, loader: StatementBankLoader, random_seed: int = None):
        """Initialize sampler with statement bank loader.
        
        Args:
            loader: Loaded StatementBankLoader instance
            random_seed: Random seed for reproducible sampling
        """
        self.loader = loader
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
    
    def sample_ntml(self, n_truths: int, n_lies: int, sample_seed: int = None) -> List[Dict]:
        """Sample N truths and M lies, then shuffle them.
        
        Args:
            n_truths: Number of truth statements to sample
            n_lies: Number of lie statements to sample
            sample_seed: Random seed for this specific sample
            
        Returns:
            List of shuffled statement dictionaries
        """
        # Use the provided sample seed for statement sampling
        # This ensures each sample gets different statements
        seed_to_use = sample_seed if sample_seed is not None else self.random_seed
        
        # Sample statements from banks with the specific seed
        truths, lies = self.loader.sample_statements(
            n_truths, n_lies, random_state=seed_to_use
        )
        
        # Combine all statements
        all_statements = truths + lies
        
        # Shuffle the combined list (also using the same seed for consistency)
        if seed_to_use is not None:
            random.seed(seed_to_use)
        random.shuffle(all_statements)
        
        logger.debug(f"Sampled and shuffled {n_truths} truths + {n_lies} lies with seed {seed_to_use}")
        
        return all_statements
    
    def generate_multiple_samples(
        self, 
        n_truths: int, 
        n_lies: int, 
        num_samples: int
    ) -> List[List[Dict]]:
        """Generate multiple samples of N truths and M lies.
        
        Args:
            n_truths: Number of truth statements per sample
            n_lies: Number of lie statements per sample
            num_samples: Number of samples to generate
            
        Returns:
            List of statement lists, each containing shuffled truths and lies
        """
        samples = []
        
        for i in range(num_samples):
            # Use a different random state for each sample to ensure variety
            # This creates a unique seed for each sample based on the base seed + sample index
            sample_seed = self.random_seed + i if self.random_seed is not None else None
            
            # Pass the sample-specific seed to get different statements for each sample
            sample_statements = self.sample_ntml(n_truths, n_lies, sample_seed)
            samples.append(sample_statements)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples for {n_truths}T{n_lies}L")
        
        return samples

class DatasetGenerator:
    """Orchestrates full NTML dataset generation pipeline."""
    
    def __init__(self, config: GenerationConfig = None):
        """Initialize generator with configuration.
        
        Args:
            config: Generation configuration, uses default if None
        """
        self.config = config or DEFAULT_CONFIG
        self.loader = None
        self.sampler = None
        self.formatter = ConversationFormatter()
        
    def initialize(self):
        """Initialize statement bank loader and sampler."""
        logger.info("Initializing NTML dataset generator...")
        
        # Load statement banks
        self.loader = StatementBankLoader(
            self.config.truth_bank_path,
            self.config.lie_bank_path
        )
        self.loader.load_banks()
        
        # Log bank statistics
        stats = self.loader.get_bank_statistics()
        logger.info(f"Bank statistics: {stats}")
        
        # Initialize sampler
        self.sampler = NTMLSampler(self.loader, self.config.random_seed)
        
        logger.info("Initialization complete")
    
    def generate_ratio_dataset(
        self, 
        ratio_config: Dict[str, Any], 
        num_samples: int = None
    ) -> List[Dict[str, Any]]:
        """Generate dataset for a specific truth/lie ratio.
        
        Args:
            ratio_config: Dictionary with 'name', 'truths', 'lies' keys
            num_samples: Number of samples to generate, uses config default if None
            
        Returns:
            List of formatted conversation dictionaries
        """
        if not self.loader or not self.sampler:
            raise RuntimeError("Generator not initialized. Call initialize() first.")
        
        ratio_name = ratio_config['name']
        n_truths = ratio_config['truths']
        n_lies = ratio_config['lies']
        num_samples = num_samples or self.config.samples_per_ratio
        
        logger.info(f"Generating {num_samples} samples for ratio {ratio_name} ({n_truths}T{n_lies}L)")
        
        # Generate multiple statement samples
        statement_samples = self.sampler.generate_multiple_samples(
            n_truths, n_lies, num_samples
        )
        
        # Format each sample as a conversation
        conversations = []
        for i, statements in enumerate(statement_samples):
            conversation_id = f"{ratio_name}_{i:03d}"
            
            # Determine lie positions (1-indexed) after statements are shuffled
            lie_positions = []
            for pos, stmt in enumerate(statements, 1):  # 1-indexed positions
                if not stmt['label']:  # label=False means it's a lie
                    lie_positions.append(pos)
            
            # Generate system prompt with specific lie positions
            system_prompt = self.config.get_system_prompt(
                n_truths, n_lies, lie_positions
            )
            
            conversation = self.formatter.format_conversation(
                system_prompt=system_prompt,
                statements=statements,
                conversation_id=conversation_id,
                add_timestamp=self.config.add_timestamps
            )
            
            # Add lie positions to metadata for analysis (always useful)
            conversation['labels']['lie_positions'] = lie_positions
            
            # Validate conversation if enabled
            if self.config.validate_output:
                if not self.formatter.validate_positions(conversation):
                    logger.error(f"Position validation failed for {conversation_id}")
                    continue
            
            conversations.append(conversation)
        
        logger.info(f"Generated {len(conversations)} valid conversations for {ratio_name}")
        return conversations
    
    def generate_all_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate datasets for all configured ratios.
        
        Returns:
            Dictionary mapping ratio names to conversation lists
        """
        if not self.loader or not self.sampler:
            raise RuntimeError("Generator not initialized. Call initialize() first.")
        
        logger.info(f"Generating datasets for {len(self.config.target_ratios)} ratios")
        
        all_datasets = {}
        
        for ratio_config in self.config.target_ratios:
            ratio_name = ratio_config['name']
            
            try:
                conversations = self.generate_ratio_dataset(ratio_config)
                all_datasets[ratio_name] = conversations
                
            except Exception as e:
                logger.error(f"Failed to generate dataset for {ratio_name}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(all_datasets)} datasets")
        return all_datasets
    
    def save_dataset(self, conversations: List[Dict[str, Any]], output_path: str):
        """Save dataset to JSONL file.
        
        Args:
            conversations: List of conversation dictionaries
            output_path: Path to output JSONL file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(conversations)} conversations to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for conversation in conversations:
                json.dump(conversation, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Dataset saved to {output_path}")
    
    def save_all_datasets(self, datasets: Dict[str, List[Dict[str, Any]]]):
        """Save all generated datasets to individual JSONL files.
        
        Args:
            datasets: Dictionary mapping ratio names to conversation lists
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for ratio_name, conversations in datasets.items():
            filename = self.config.get_output_filename(ratio_name)
            output_path = output_dir / filename
            self.save_dataset(conversations, output_path)
    
    def generate_and_save_all(self):
        """Complete pipeline: initialize, generate all datasets, and save them."""
        logger.info("Starting complete NTML dataset generation pipeline")
        
        # Initialize components
        self.initialize()
        
        # Generate all datasets
        datasets = self.generate_all_datasets()
        
        # Save all datasets
        self.save_all_datasets(datasets)
        
        # Log summary
        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(f"Pipeline complete: generated {total_conversations} total conversations")
        
        return datasets


def generate_ntml_datasets(config: GenerationConfig = None) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function to generate NTML datasets.
    
    Args:
        config: Generation configuration, uses default if None
        
    Returns:
        Dictionary mapping ratio names to conversation lists
    """
    generator = DatasetGenerator(config)
    return generator.generate_and_save_all() 