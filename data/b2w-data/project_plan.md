# Back-to-Work Data Integration Project Plan

## Project Overview
Integrate previous back-to-work (B2W) pipeline data with the new probe-based analysis system from Probity. This involves data cleanup, format standardization, and implementing a configurable pipeline for probe score augmentation.

## Goals
1. Create a unified pipeline that can:
   - Take arbitrary input datasets
   - Apply trained probes from Probity
   - Augment data with probe scores
   - Save in a specified format/location

2. Successfully integrate B2W data with Probity's probe analysis
   - Clean and standardize old B2W data
   - Apply new NTML probes to B2W data
   - Generate comparable metrics

## B2W Data Format
The input data is in JSONL format, with each line containing a JSON object with the following structure:

```json
{
  // Metadata
  "dataset": "ai_liar",              // Dataset identifier
  "sample_index": 1,                 // Index in dataset
  "label": "deceptive",             // Conversation label

  // Conversation Data
  "messages": [                      
    {
      "role": "user/assistant",      // Message role
      "content": "..."               // Message content
    }
  ],

  // Token Analysis
  "token_analysis": {
    "tokens": [...],                 // Tokenized text
    "public_indices": [...],         // Public token indices
    "private_indices": [...],        // Private token indices
    "critical_indices": [...]        // Critical token indices (can be null)
  },

  // Monitor Results
  "monitor_results": [
    {
      "monitor": {
        "name": "mean_pairs",        // Monitor identifier
        "method": "full_averaging",  // Analysis method
        "probe": "pairs_probe",      // Probe type used
        "model": "no_model",         // Model used (if any)
        "metadata": null             // Additional metadata
      },
      "result": {
        "suspiciousness": -0.52,     // Suspiciousness score
        "reasoning": {               // Reasoning details
          "type": "mean",            // Reasoning type
          "num_scores": 232          // Number of scores used
        },
        "metadata": {}               // Additional metadata
      },
      "timestamp": "..."            // Analysis timestamp
    }
  ]
}
```

### Monitor Types
1. **Averaging Methods**
   - full_averaging: Average across all tokens
   - critical_averaging: Average across critical tokens only
   - peak_score: Highest score found

2. **Reasoning Methods**
   - black_box_reasoning: Model-based analysis without probe scores
   - white_box_reasoning: Model-based analysis with probe scores

3. **Probe Types**
   - pairs_probe: Pairwise comparison probe
   - rp_probe: Response prediction probe
   - no_probe: Direct model analysis

## Implementation Steps

### Phase 1: Data Preparation
1. [ ] Review and document old B2W data format
2. [ ] Define target data format for probe analysis
3. [ ] Create data cleaning/conversion scripts
4. [ ] Validate converted data format

### Phase 2: Probe Integration
1. [ ] Create configurable probe loading system
2. [ ] Implement efficient batch processing
3. [ ] Define score augmentation format
4. [ ] Add visualization capabilities

### Phase 3: Pipeline Development
1. [ ] Create main script with configurable:
   - Input data path/format
   - Probe selection
   - Model selection
   - Output path/format
2. [ ] Implement efficient caching
3. [ ] Add progress tracking
4. [ ] Add error handling and validation

### Phase 4: Testing & Validation
1. [ ] Test with small B2W dataset
2. [ ] Validate probe scores
3. [ ] Compare with previous results
4. [ ] Document performance metrics

## Directory Structure
```
data/
├── b2w-data/      # Input data directory
│   ├── raw/       # Original B2W data
│   └── cleaned/   # Processed data ready for probe analysis
└── b2w-scores/    # Output directory
    ├── raw/       # Raw probe scores
    └── processed/ # Final augmented datasets
```

## Technical Requirements
1. Data Format Requirements:
   - Input: JSONL with messages, token analysis, and optional monitor results
   - Output: JSONL with augmented probe scores and analysis
   - Support for multiple monitor types and reasoning methods

2. Performance Requirements:
   - Efficient batch processing
   - Activation caching
   - Memory management for large datasets

3. Integration Requirements:
   - Compatible with Probity's probe system
   - Maintainable and extensible
   - Well-documented API

## Next Steps
1. [ ] Document old B2W data format
2. [ ] Create sample of cleaned data format
3. [ ] Begin implementation of data cleaning pipeline 