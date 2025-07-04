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
  "dataset": "alpaca__plain",          // Dataset identifier
  "sample_index": 1,                   // Index in dataset
  "label": "honest",                   // Conversation label

  // Conversation Data (Input for Probe Scoring)
  "messages": [                        
    {
      "role": "system/user/assistant", 
      "content": "..."                 
    }
  ],

  // Existing Analysis Data
  "tokens": [...],                     // Tokenized text
  "probe_scores": {                    // Existing probe scores
    "pairs_probe": [...],              
    "rp_probe": [...]                  
  },
  "public_assistant_indices": [...],    
  "public_user_indices": [...],         
  "public_messages": [...],             
  "critical_analysis": {...}            
}
```

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
   - TBD: Document B2W data format
   - TBD: Document probe score format
   - TBD: Document output format

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