#!/bin/bash
set -e

echo "🚀 Setting up Soothcheck environment on RunPod..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Create and activate conda environment
print_status "Creating conda environment 'soothcheck'..."
conda create -n soothcheck python=3.10 -y

print_status "Activating conda environment..."
conda activate soothcheck

# 2. Auto-load environment for future sessions
print_status "Setting up auto-activation for future sessions..."
if ! grep -q "conda activate soothcheck" ~/.bashrc; then
    echo 'conda activate soothcheck' >> ~/.bashrc
    print_success "Added conda activation to .bashrc"
else
    print_warning "Conda activation already in .bashrc"
fi

# 3. Install PyTorch with CUDA support
print_status "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install Hugging Face and core dependencies
print_status "Installing Hugging Face and core dependencies..."
pip install transformers>=4.30.0
pip install huggingface_hub
pip install accelerate
pip install bitsandbytes  # For quantization support
pip install sentencepiece  # For tokenizers
pip install protobuf  # For model loading

# 5. Install other dependencies
print_status "Installing additional dependencies..."
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install transformer_lens==2.15.4
pip install scikit-learn>=1.3.0
pip install datasets>=2.12.0
pip install tqdm>=4.65.0
pip install tabulate>=0.9.0
pip install pandas
pip install seaborn

# 6. Install the probity package in development mode
print_status "Installing probity package in development mode..."
pip install -e .

# 7. Verify installation
print_status "Verifying installation..."
python -c "
import torch
import transformers
import huggingface_hub
import probity

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
print(f'Transformers version: {transformers.__version__}')
print(f'Hugging Face Hub version: {huggingface_hub.__version__}')
print('All imports successful!')
"

# 8. Test GPU memory
if torch.cuda.is_available(); then
    print_status "Testing GPU memory..."
    python -c "
import torch
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    print(f'Device {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"
fi

print_success "🎉 Setup complete!"
print_status "To activate the environment in future sessions:"
echo "  conda activate soothcheck"
echo ""
print_status "To test the setup, run:"
echo "  PYTHONPATH=\$(pwd) python test_ntml_gpt2.py"
echo ""
print_status "For multi-GPU testing, modify the MultiGPUConfig in your scripts:"
echo "  multi_gpu = MultiGPUConfig(enabled=True, device_ids=[0, 1])" 