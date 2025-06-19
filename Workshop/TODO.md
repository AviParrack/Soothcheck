- long sequence training without truncation (likely needed at data processing step)
"For true long-form SFT where individual documents are much larger (e.g., 10,000 tokens), you would implement chunking with overlap in scripts/prepare_dataset.py. Each chunk would become a separate entry in your processed dataset."

