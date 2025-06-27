# LLM GitIngest - Repository Structure Reference

This file provides a comprehensive overview of the Soothcheck repository structure for easy reference during development and analysis.

## Directory Structure

```
└── aviparrack-soothcheck.git/
    ├── README.md
    ├── generate_ntml_datasets.py
    ├── LICENSE
    ├── LLMagents.md
    ├── mac_test_script.py
    ├── probity-integration-analysis.md
    ├── pyproject.toml
    ├── soothcheck_probe_training.ipynb
    ├── Statement-level-probe-implementation-plan.md
    ├── data/
    │   ├── two_truths_one_lie.json
    │   ├── two_truths_one_lie_eval.json
    │   ├── two_truths_one_lie_eval_with_fullstops.json
    │   ├── two_truths_one_lie_tiny.json
    │   ├── two_truths_one_lie_with_fullstops.json
    │   ├── NTML-datasets/
    │   │   ├── 2T1L_20samples.jsonl
    │   │   ├── 2T1L_2samples.jsonl
    │   │   ├── 2T1L_3samples.jsonl
    │   │   ├── 5T5L_3samples.jsonl
    │   │   ├── 5T5L_5samples.jsonl
    │   │   ├── NTML-module-README.md
    │   │   └── generation-scripts/
    │   │       ├── __init__.py
    │   │       ├── config.py
    │   │       ├── formatters.py
    │   │       ├── generate_datasets.py
    │   │       ├── ntml_generator.py
    │   │       └── statement_loader.py
    │   └── statement-banks/
    │       ├── README.md
    │       ├── lie_bank.csv
    │       └── truth_bank.csv
    ├── debate/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── types.py
    │   ├── analysis/
    │   │   ├── __init__.py
    │   │   ├── metrics.py
    │   │   ├── probe_analysis.py
    │   │   └── visualization.py
    │   ├── inference/
    │   │   ├── __init__.py
    │   │   ├── activation_cache.py
    │   │   └── probe_debate_inference.py
    │   ├── orchestration/
    │   │   ├── __init__.py
    │   │   ├── conversation.py
    │   │   ├── debate_manager.py
    │   │   └── formats.py
    │   └── providers/
    │       ├── __init__.py
    │       ├── anthropic.py
    │       ├── base.py
    │       ├── local.py
    │       ├── openai.py
    │       ├── openrouter.py
    │       └── replicate.py
    ├── probity/
    │   ├── __init__.py
    │   ├── probity-README.md
    │   ├── analysis/
    │   │   ├── __init__.py
    │   │   └── ttol_analysis.py
    │   ├── collection/
    │   │   ├── __init__.py
    │   │   ├── activation_store.py
    │   │   └── collectors.py
    │   ├── datasets/
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── position_finder.py
    │   │   ├── templated.py
    │   │   └── tokenized.py
    │   ├── evaluation/
    │   │   ├── __init__.py
    │   │   └── batch_evaluator.py
    │   ├── pipeline/
    │   │   └── pipeline.py
    │   ├── probes/
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── config.py
    │   │   ├── directional.py
    │   │   ├── inference.py
    │   │   ├── linear.py
    │   │   ├── logistic.py
    │   │   ├── probe_set.py
    │   │   └── sklearn_logistic.py
    │   ├── training/
    │   │   ├── __init__.py
    │   │   ├── configs.py
    │   │   └── trainer.py
    │   ├── utils/
    │   │   ├── __init__.py
    │   │   ├── caching.py
    │   │   └── dataset_loading.py
    │   └── visualisation/
    │       ├── probe_performance.py
    │       ├── token_highlight.py
    │       └── ttol_visualisation.py
    ├── probity_extensions/
    │   ├── __init__.py
    │   └── conversational.py
    ├── scripts/
    │   ├── add_2t1l_analysis.py
    │   ├── batch_debate_analysis.py
    │   ├── debate_runner.py
    │   ├── probe_eval.py
    │   ├── probe_training.py
    │   ├── test_ntml_probity_integration.py
    │   ├── train_ntml_probes.py
    │   └── utils.py
    ├── tests/
    │   ├── test_probe_save_load.py
    │   └── unit/
    │       ├── datasets/
    │       │   ├── test_base.py
    │       │   ├── test_position_finder.py
    │       │   ├── test_templated.py
    │       │   └── test_tokenized.py
    │       ├── pipeline/
    │       │   └── test_pipeline.py
    │       └── trainer/
    │           └── test_trainer.py
    ├── trained_probes/
    │   ├── ntml_2T1L_2samples_blocks_7_hook_resid_pre.pt
    │   ├── training_summary.json
    │   ├── linear/
    │   │   ├── layer_0_probe.json
    │   │   ├── layer_10_probe.json
    │   │   ├── layer_11_probe.json
    │   │   ├── layer_12_probe.json
    │   │   ├── layer_13_probe.json
    │   │   ├── layer_14_probe.json
    │   │   ├── layer_15_probe.json
    │   │   ├── layer_16_probe.json
    │   │   ├── layer_17_probe.json
    │   │   ├── layer_18_probe.json
    │   │   ├── layer_19_probe.json
    │   │   ├── layer_1_probe.json
    │   │   ├── layer_20_probe.json
    │   │   ├── layer_21_probe.json
    │   │   ├── layer_22_probe.json
    │   │   ├── layer_23_probe.json
    │   │   ├── layer_24_probe.json
    │   │   ├── layer_25_probe.json
    │   │   ├── layer_26_probe.json
    │   │   ├── layer_27_probe.json
    │   │   ├── layer_28_probe.json
    │   │   ├── layer_29_probe.json
    │   │   ├── layer_2_probe.json
    │   │   ├── layer_30_probe.json
    │   │   ├── layer_31_probe.json
    │   │   ├── layer_3_probe.json
    │   │   ├── layer_4_probe.json
    │   │   ├── layer_5_probe.json
    │   │   ├── layer_6_probe.json
    │   │   ├── layer_7_probe.json
    │   │   ├── layer_8_probe.json
    │   │   └── layer_9_probe.json
    │   ├── logistic/
    │   │   ├── layer_0_probe.json
    │   │   ├── layer_10_probe.json
    │   │   ├── layer_11_probe.json
    │   │   ├── layer_12_probe.json
    │   │   ├── layer_13_probe.json
    │   │   ├── layer_14_probe.json
    │   │   ├── layer_15_probe.json
    │   │   ├── layer_16_probe.json
    │   │   ├── layer_17_probe.json
    │   │   ├── layer_18_probe.json
    │   │   ├── layer_19_probe.json
    │   │   ├── layer_1_probe.json
    │   │   ├── layer_20_probe.json
    │   │   ├── layer_21_probe.json
    │   │   ├── layer_22_probe.json
    │   │   ├── layer_23_probe.json
    │   │   ├── layer_24_probe.json
    │   │   ├── layer_25_probe.json
    │   │   ├── layer_26_probe.json
    │   │   ├── layer_27_probe.json
    │   │   ├── layer_28_probe.json
    │   │   ├── layer_29_probe.json
    │   │   ├── layer_2_probe.json
    │   │   ├── layer_30_probe.json
    │   │   ├── layer_31_probe.json
    │   │   ├── layer_3_probe.json
    │   │   ├── layer_4_probe.json
    │   │   ├── layer_5_probe.json
    │   │   ├── layer_6_probe.json
    │   │   ├── layer_7_probe.json
    │   │   ├── layer_8_probe.json
    │   │   └── layer_9_probe.json
    │   ├── meandiff/
    │   │   ├── layer_0_probe.json
    │   │   ├── layer_10_probe.json
    │   │   ├── layer_11_probe.json
    │   │   ├── layer_12_probe.json
    │   │   ├── layer_13_probe.json
    │   │   ├── layer_14_probe.json
    │   │   ├── layer_15_probe.json
    │   │   ├── layer_16_probe.json
    │   │   ├── layer_17_probe.json
    │   │   ├── layer_18_probe.json
    │   │   ├── layer_19_probe.json
    │   │   ├── layer_1_probe.json
    │   │   ├── layer_20_probe.json
    │   │   ├── layer_21_probe.json
    │   │   ├── layer_22_probe.json
    │   │   ├── layer_23_probe.json
    │   │   ├── layer_24_probe.json
    │   │   ├── layer_25_probe.json
    │   │   ├── layer_26_probe.json
    │   │   ├── layer_27_probe.json
    │   │   ├── layer_28_probe.json
    │   │   ├── layer_29_probe.json
    │   │   ├── layer_2_probe.json
    │   │   ├── layer_30_probe.json
    │   │   ├── layer_31_probe.json
    │   │   ├── layer_3_probe.json
    │   │   ├── layer_4_probe.json
    │   │   ├── layer_5_probe.json
    │   │   ├── layer_6_probe.json
    │   │   ├── layer_7_probe.json
    │   │   ├── layer_8_probe.json
    │   │   └── layer_9_probe.json
    │   └── pca/
    │       ├── layer_0_probe.json
    │       ├── layer_10_probe.json
    │       ├── layer_11_probe.json
    │       ├── layer_12_probe.json
    │       ├── layer_13_probe.json
    │       ├── layer_14_probe.json
    │       ├── layer_15_probe.json
    │       ├── layer_16_probe.json
    │       ├── layer_17_probe.json
    │       ├── layer_18_probe.json
    │       ├── layer_19_probe.json
    │       ├── layer_1_probe.json
    │       ├── layer_20_probe.json
    │       ├── layer_21_probe.json
    │       ├── layer_22_probe.json
    │       ├── layer_23_probe.json
    │       ├── layer_24_probe.json
    │       ├── layer_25_probe.json
    │       ├── layer_26_probe.json
    │       ├── layer_27_probe.json
    │       ├── layer_28_probe.json
    │       ├── layer_29_probe.json
    │       ├── layer_2_probe.json
    │       ├── layer_30_probe.json
    │       ├── layer_31_probe.json
    │       ├── layer_3_probe.json
    │       ├── layer_4_probe.json
    │       ├── layer_5_probe.json
    │       ├── layer_6_probe.json
    │       ├── layer_7_probe.json
    │       ├── layer_8_probe.json
    │       └── layer_9_probe.json
    └── tutorials/
        ├── 1-probity-basics.py
        ├── 2-dataset-creation.py
        ├── 3-probe-variants.py
        └── 4-multiclass-probe.py
```

## Key Components Summary

### Core Modules
- **`probity/`** - Main probing library for neural activation analysis
- **`debate/`** - Debate orchestration and analysis framework  
- **`probity_extensions/`** - Extensions for conversational probing (NTML integration)

### Data & Datasets
- **`data/NTML-datasets/`** - Neural Truth-Finding & Lie-Detection datasets in JSONL format
- **`data/statement-banks/`** - Truth and lie statement repositories

### Scripts & Tools
- **`scripts/train_ntml_probes.py`** - NTML probe training with statement-level analysis
- **`mac_test_script.py`** - Lightweight testing script for Mac development
- **`generate_ntml_datasets.py`** - Dataset generation utility

### Trained Models
- **`trained_probes/`** - Contains trained probes organized by type (linear, logistic, meandiff, pca)
- Each probe type has models for layers 0-31 of transformer architectures

### Analysis & Evaluation
- **`probity/evaluation/`** - Batch evaluation tools for probe performance assessment
- **`probity/visualisation/`** - Visualization tools for probe performance and token analysis
- **`debate/analysis/`** - Metrics and analysis for debate scenarios

## Recent Improvements
- **Last Token Positioning**: Fixed statement probing to use last token instead of first token for autoregressive models
- **NTML Integration**: Seamless integration between NTML dataset generation and probity training
- **Enhanced Debugging**: Comprehensive debug output with `--probe-debug` flag
- **Mac Compatibility**: Lightweight testing with GPT-2 instead of heavy Llama models

This structure enables comprehensive truth detection research through neural activation probing and conversational analysis. 