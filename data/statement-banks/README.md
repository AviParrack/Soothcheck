# Statement Banks

This directory contains balanced statement banks created for truth detection research in the Soothcheck project.

## Data Source

The statement banks were created by combining, balancing, and separating by label the datasets from [@balevinstein/Probes](https://github.com/balevinstein/Probes.git). The original datasets in that repository were used for supervised and unsupervised probing research, largely based on Azaria and Mitchell's work "The Internal State of an LLM Knows When It's Lying."

## Files

- **`truth_bank.csv`** - 3,968 true statements extracted and balanced from the original datasets
- **`lie_bank.csv`** - 3,968 false statements extracted and balanced from the original datasets

## Processing

The original mixed-statement datasets were processed to:
1. Separate statements by their truth labels
2. Balance the number of true and false statements
3. Create dedicated banks for sampling in NTML (N Truths, M Lies) dataset generation

## Original Dataset Categories

The statements cover diverse domains including:
- Animals and biological facts
- Companies and business information  
- Cities and geographical data
- Historical events and figures
- Scientific facts and phenomena
- And other factual domains

## Usage

These statement banks serve as the foundation for generating conversational datasets with controlled truth/lie ratios for statement-level truth detection research. They are used by the NTML dataset generation pipeline in `../NTML-datasets/`.

## Attribution

Original datasets sourced from the [Probes repository](https://github.com/balevinstein/Probes.git) by balevinstein, which implements probing methods for large language models based on academic research in truthfulness detection. 