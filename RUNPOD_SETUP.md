# 🚀 RunPod Setup Guide: NTML Probing with Llama 3.3 70B Instruct

This guide will get you up and running with statement-level probing on Llama 3.3 70B Instruct in RunPod within minutes.

## 📋 Prerequisites

### RunPod Configuration
- **GPU:** H100 (80GB) or A100 (80GB) recommended for full precision
- **Alternative:** RTX 6000 Ada (48GB) with 8-bit quantization
- **Template:** PyTorch 2.0+ with CUDA 12.1+
- **Storage:** 200GB+ persistent volume (**CRITICAL**: Llama 3.3 70B is ~140GB)
- **RAM:** 64GB+ system memory
- **Network:** Stable high-speed connection (model download takes 20-60 minutes)

### Required Tokens
- **HuggingFace Token:** For Llama 3.3 70B access (set as `HF_TOKEN` environment variable)

### Optional Tokens  
- **OpenAI API Key:** Only needed if you want to generate custom NTML datasets with OpenAI models (not required for setup or basic usage)

### ⚠️ Important: Model Download
**The setup script will automatically download Llama 3.3 70B (~140GB)**:
- First run takes 20-60 minutes for download
- Requires 150GB+ free storage space
- Model is cached for future use
- Download happens once per RunPod instance

### Model Compliance
This setup follows **Probity library standards**:
- Uses `HookedTransformer.from_pretrained_no_processing()` for model loading
- Implements `torch.bfloat16` dtype (standard for Llama models in Probity)
- Compatible with existing Probity pipeline and probe architectures
- Extends Probity with conversational dataset support

## 🚀 One-Command Setup

**Option 1: With HF_TOKEN environment variable**
```bash
# Set your token first, then run setup
export HF_TOKEN="your_huggingface_token_here"
curl -sSL https://raw.githubusercontent.com/AviParrack/Soothcheck/main/probity/runpod_setup.sh | bash
```

**Option 2: Interactive login during setup**
```bash
# Setup will prompt you to login interactively
curl -sSL https://raw.githubusercontent.com/AviParrack/Soothcheck/main/probity/runpod_setup.sh | bash
# When prompted, run: huggingface-cli login
```

## 🔧 Manual Setup (Step by Step)

### 1. Launch RunPod Instance & Setup Authentication

**Choose one of these authentication methods:**

**Method A: Environment Variable (Recommended)**
```bash
# Set your HuggingFace token
export HF_TOKEN="your_huggingface_token_here"

# Verify it's set
echo $HF_TOKEN
```

**Method B: Interactive Login**
```bash
# Login interactively (will prompt for token)
huggingface-cli login
```

**Method C: Manual Token File**
```bash
# Create token file manually
echo "your_huggingface_token_here" > ~/.cache/huggingface/token
```

### 2. Verify Llama 3.3 Access
```bash
# Test access before running setup
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')"
```

### 3. Clone and Install
```bash
# Clone repository
git clone https://github.com/AviParrack/Soothcheck.git Jord-probity
cd Jord-probity/probity

# Install dependencies
pip install -e .
pip install accelerate bitsandbytes transformers>=4.45.0  # For Llama 3.3 70B
```

### 4. Verify Installation
```bash
# Run environment check
python check_llama33_environment.py

# Test NTML integration
python scripts/test_ntml_probity_integration.py --verbose
```

### 5. Generate Test Dataset
```bash
# Create a small test dataset
python data/NTML-datasets/generate_datasets.py \
    --dataset_name "llama33_test" \
    --num_conversations 5 \
    --truths 3 \
    --lies 2 \
    --output_dir data/NTML-datasets/
```

### 6. Train Your First Probe
```bash
# Train probe on Llama 3.3 70B (adjust based on GPU memory)

# For H100/A100 80GB (full precision):
python scripts/train_ntml_probes.py \
    --dataset llama33_test \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --hook_point blocks.40.hook_resid_pre \
    --debug

# For RTX 6000 Ada 48GB (8-bit quantization):
python scripts/train_ntml_probes.py \
    --dataset llama33_test \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --hook_point blocks.40.hook_resid_pre \
    --load_in_8bit \
    --debug
```

## 🎯 Quick Start Commands

### Check System Status
```bash
python check_llama33_environment.py
```

### List Available Datasets
```bash
python scripts/train_ntml_probes.py --list-datasets
```

### Run Quick Start Demo
```bash
python quick_start_llama33_70b.py
```

### Train Custom Probe
```bash
python scripts/train_ntml_probes.py \
    --dataset <dataset_name> \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --hook_point blocks.<layer>.hook_resid_pre \
    --debug
```

## 📊 Understanding the Output

### Training Progress
```
🏋️ Training NTML probe: llama33_test_blocks_40_hook_resid_pre
Dataset: llama33_test (15 statement examples from 5 conversations)
Model: meta-llama/Llama-3.3-70B-Instruct
Hook: blocks.40.hook_resid_pre

Epoch 1/10: Train Loss: 0.6234, Val Loss: 0.5891
Epoch 2/10: Train Loss: 0.4123, Val Loss: 0.4567
...
✅ Training complete! Probe saved to: trained_probes/llama33_test_blocks_40_hook_resid_pre.pt
```

### Debug Mode Insights
With `--debug`, you'll see:
- **Conversation Structure:** System prompts with lie positions
- **Statement Extraction:** How statements are parsed and labeled
- **Tokenization:** Token positions for each statement
- **Activation Collection:** Model processing and hook extraction
- **Training Data:** Exact examples used for probe training

## 🔍 Key Directories

```
probity/
├── data/NTML-datasets/          # Your NTML conversation datasets
├── trained_probes/              # Saved probe models
├── cache/ntml_cache/           # Cached model activations
├── probity_extensions/         # NTML-specific extensions
└── scripts/                    # Training and testing scripts
```

## 🧪 Advanced Usage

### Multi-Layer Analysis
```bash
# Train probes on multiple layers (Llama 3.3 70B has 80 layers)
for layer in 20 40 60; do
    python scripts/train_ntml_probes.py \
        --dataset llama33_test \
        --model_name meta-llama/Llama-3.3-70B-Instruct \
        --hook_point blocks.${layer}.hook_resid_pre
done
```

### Custom Dataset Generation
```bash
# Generate larger dataset with specific parameters
python data/NTML-datasets/generate_datasets.py \
    --dataset_name "custom_experiment" \
    --num_conversations 50 \
    --truths 5 \
    --lies 3 \
    --topics "science,history,geography" \
    --output_dir data/NTML-datasets/
```

### Memory Optimization
```bash
# Use gradient checkpointing for memory efficiency
python scripts/train_ntml_probes.py \
    --dataset llama33_test \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --hook_point blocks.40.hook_resid_pre \
    --gradient_checkpointing \
    --batch_size 2
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use quantization
--load_in_8bit  # or --load_in_4bit for extreme memory constraints

# Reduce batch size
--batch_size 1

# Set environment variable
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**2. HuggingFace Token Issues**
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token directly
export HF_TOKEN="your_token_here"

# Check access
python -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct'))"
```

**3. Model Loading Errors**
```bash
# Verify TransformerLens compatibility
python -c "from transformer_lens import HookedTransformer; print('TransformerLens available')"

# Check Probity compliance
python -c "from probity.collection.collectors import TransformerLensCollector; print('Probity ready')"
```

**4. Import Errors**
```bash
# Reinstall dependencies
pip install -e . --force-reinstall
pip install accelerate bitsandbytes transformers>=4.45.0
```

### Performance Optimization

**For H100 80GB (Optimal):**
```bash
python scripts/train_ntml_probes.py \
    --dataset llama33_test \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --hook_point blocks.40.hook_resid_pre \
    --batch_size 4
```

**For A100 80GB (Good):**
```bash
python scripts/train_ntml_probes.py \
    --dataset llama33_test \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --hook_point blocks.40.hook_resid_pre \
    --load_in_8bit \
    --batch_size 2
```

**For RTX 6000 Ada 48GB (Marginal):**
```bash
python scripts/train_ntml_probes.py \
    --dataset llama33_test \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --hook_point blocks.40.hook_resid_pre \
    --load_in_8bit \
    --batch_size 1 \
    --gradient_checkpointing
```

## 📈 Model Architecture Details

### Llama 3.3 70B Specifications
- **Parameters:** 70.6B
- **Layers:** 80 (blocks.0 to blocks.79)
- **Hidden Size:** 8192
- **Attention Heads:** 64
- **Context Length:** 128k tokens
- **Dtype:** `torch.bfloat16` (Probity standard)

### Recommended Hook Points
- **Early layers:** `blocks.20.hook_resid_pre` - Basic representations
- **Middle layers:** `blocks.40.hook_resid_pre` - Complex reasoning
- **Late layers:** `blocks.60.hook_resid_pre` - Final processing

## 🎉 Success Indicators

You'll know everything is working when you see:
- ✅ GPU detected with sufficient VRAM (80GB+ recommended)
- ✅ Llama 3.3 70B access confirmed via HuggingFace
- ✅ TransformerLens compatibility verified
- ✅ NTML datasets generated successfully
- ✅ Probe training completes without OOM errors
- ✅ Trained probes saved to `trained_probes/`

## 📞 Support

If you encounter issues:
1. Check the debug output with `--debug` flag
2. Verify GPU memory with `nvidia-smi`
3. Run `python check_llama33_environment.py` for diagnostics
4. Review the conversation structure in your NTML datasets
5. Consider using quantization if memory is limited

## 🔬 Research Notes

This implementation solves **Carlo's causal attention problem** by:
1. **Enhanced System Prompts:** Specify lie positions upfront
2. **Statement-Level Probing:** Extract activations at exact token positions
3. **Full Context Processing:** Model sees entire conversation including lie positions
4. **Probity Integration:** 90% code reuse with existing pipeline

---

**🚀 Ready to discover how Llama 3.3 70B represents truthfulness at the statement level!** 