#!/bin/bash
set -e  # Exit on any error

echo "=== ML Environment Setup ==="
echo "This script is idempotent - it's safe to run multiple times"

# Detect platform
if [[ $(uname -m) == "arm64" ]] && [[ $(uname) == "Darwin" ]]; then
  PLATFORM="macos-arm64"
  echo "Detected platform: Apple Silicon MacBook"
else
  PLATFORM="linux-x86_64"
  echo "Detected platform: Linux x86_64 (vast.ai or similar)"
fi

echo "Step 1: Checking Conda installation..."
CONDA_NEEDS_INSTALLATION=false
MINICONDA_INSTALL_PATH="$HOME/miniconda3" # Default install path if we need to install it

if command -v conda &> /dev/null; then
    echo "Conda command is already available. Skipping Miniconda installation."
    CONDA_BASE_PATH=$(conda info --base)
    echo "Using existing Conda at $CONDA_BASE_PATH"
    # Ensure conda is initialized for the current script session
    if [ -z "$CONDA_SHLVL" ] || [ "$CONDA_SHLVL" -lt 1 ]; then # Check if Conda is already activated/initialized in this shell
      if [ -f "$CONDA_BASE_PATH/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
        echo "Initialized Conda from $CONDA_BASE_PATH for the current script session."
      else
        echo "Warning: Could not find conda.sh in existing Conda at $CONDA_BASE_PATH/etc/profile.d/conda.sh. Script might face issues with conda commands."
      fi
    else
        echo "Conda already initialized in current shell (CONDA_SHLVL=$CONDA_SHLVL)."
    fi
else
    echo "Conda command not found. Installing Miniconda to $MINICONDA_INSTALL_PATH..."
    CONDA_NEEDS_INSTALLATION=true
    if [[ "$PLATFORM" == "macos-arm64" ]]; then
        # macOS ARM installer
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -O miniconda.sh
    else
        # Linux x86_64 installer
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    fi
    bash miniconda.sh -b -p "$MINICONDA_INSTALL_PATH"
    rm miniconda.sh
    echo "Miniconda installed successfully to $MINICONDA_INSTALL_PATH."
    # Initialize Conda for the current script session after installation
    export PATH="$MINICONDA_INSTALL_PATH/bin:$PATH"
    source "$MINICONDA_INSTALL_PATH/etc/profile.d/conda.sh"
    echo "Initialized newly installed Conda from $MINICONDA_INSTALL_PATH for the current script session."
fi

# Step 2: Configure shell for Conda, if it was installed by this script.
if [ "$CONDA_NEEDS_INSTALLATION" = true ]; then
    echo "Step 2: Configuring shell for the new Miniconda installation at $MINICONDA_INSTALL_PATH..."
    # Determine the shell name (e.g., bash, zsh)
    SHELL_NAME=$(basename "$SHELL")
    # Check if the conda executable exists before trying to run init
    if [ -x "$MINICONDA_INSTALL_PATH/bin/conda" ]; then
        echo "Running 'conda init $SHELL_NAME' for the Miniconda installation at $MINICONDA_INSTALL_PATH..."
        "$MINICONDA_INSTALL_PATH/bin/conda" init "$SHELL_NAME"
        echo "Conda initialization complete for $SHELL_NAME. You may need to source your shell config file (e.g., .bashrc, .zshrc) or restart your shell."
    else
        echo "Error: Could not find conda executable at $MINICONDA_INSTALL_PATH/bin/conda. Shell initialization skipped."
    fi
else
    echo "Step 2: Conda was pre-existing. Assuming shell is already configured for future logins. Skipping modification."
fi

# Detect CUDA on Linux systems
if [[ "$PLATFORM" == "linux-x86_64" ]]; then
    echo "Step 3: Detecting CUDA version..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
        echo "Detected CUDA $CUDA_VERSION"
    else
        echo "CUDA not found, will use Conda's default"
        CUDA_VERSION="11.8"  # Fallback version
        CUDA_MAJOR=11
        CUDA_MINOR=8
    fi
else
    echo "Step 3: On Apple Silicon, using Metal instead of CUDA"
fi

# Create or update environment (idempotent via existence check)
ENV_EXISTS=$(conda env list | grep -q "^ml " && echo "yes" || echo "no")
if [ "$ENV_EXISTS" == "no" ]; then
    echo "Step 4: Creating new 'ml' environment with Python 3.12..."
    conda create -y -n ml python=3.12
    echo "Environment created."
else
    echo "Step 4: Environment 'ml' exists, updating if needed."
    # Check Python version in existing env to avoid unnecessary reinstall
    PYTHON_VERSION=$(conda run -n ml python --version 2>&1 | grep -oP '(?<=Python )[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
    if [[ "$PYTHON_VERSION" == "unknown" ]]; then
        # Handle case where grep -P is not available (macOS)
        PYTHON_VERSION=$(conda run -n ml python --version 2>&1 | sed -n 's/Python \([0-9]\+\.[0-9]\+\.[0-9]\+\)/\1/p')
    fi
    PYTHON_MAJOR_MINOR=$(echo $PYTHON_VERSION | cut -d. -f1,2)
    if [ "$PYTHON_MAJOR_MINOR" != "3.12" ]; then
        echo "Python version in 'ml' is $PYTHON_VERSION, updating to 3.12..."
        echo "Skipping Python update to 3.12"
        # conda install -n ml -y python=3.12
        # echo "Python updated to 3.12"
    else
        echo "Python 3.12 already installed in 'ml' environment, skipping."
    fi
fi

# Activate the environment for further operations
echo "Installing packages in 'ml' environment..."
# Note: Instead of activating, we'll use 'conda install -n ml' and 'conda run -n ml pip'
# which work reliably in scripts

# Check if PyTorch is already installed
echo "Step 5: Checking PyTorch installation..."
PYTORCH_INSTALLED=$(conda run -n ml python -c "import torch; print('YES')" 2>/dev/null || echo "NO")

if [ "$PYTORCH_INSTALLED" != "YES" ]; then
    echo "Installing PyTorch..."
    
    if [[ "$PLATFORM" == "macos-arm64" ]]; then
        # Install PyTorch for M1/M2/M3 Macs with MPS support
        conda install -n ml -y pytorch torchvision torchaudio -c pytorch
        echo "Installed PyTorch with Metal Performance Shaders (MPS) support for Apple Silicon"
    else
        # Install PyTorch with appropriate CUDA version for Linux
        if (( CUDA_MAJOR >= 12 )); then
            # For CUDA 12.x
            conda install -n ml -y pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
        elif (( CUDA_MAJOR == 11 && CUDA_MINOR >= 8 )); then
            # For CUDA 11.8+
            conda install -n ml -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
        elif (( CUDA_MAJOR == 11 )); then
            # For CUDA 11.x below 11.8
            conda install -n ml -y pytorch torchvision torchaudio cudatoolkit=$CUDA_VERSION -c pytorch
        elif (( CUDA_MAJOR == 10 )); then
            # For CUDA 10.x
            conda install -n ml -y pytorch torchvision torchaudio cudatoolkit=$CUDA_VERSION -c pytorch
        else
            # Fallback for other versions
            conda install -n ml -y pytorch torchvision torchaudio cpuonly -c pytorch
            echo "WARNING: Installed PyTorch without CUDA support due to unrecognized CUDA version"
        fi
    fi
    echo "PyTorch installation complete."
else
    echo "PyTorch already installed, skipping."
    
    # Verify GPU support on respective platform
    if [[ "$PLATFORM" == "macos-arm64" ]]; then
        MPS_AVAILABLE=$(conda run -n ml python -c "import torch; print('YES' if torch.backends.mps.is_available() else 'NO')" 2>/dev/null || echo "NO")
        if [ "$MPS_AVAILABLE" == "YES" ]; then
            echo "MPS (Metal Performance Shaders) is available for GPU acceleration"
        else
            echo "WARNING: MPS not available. You may need to update macOS or PyTorch"
        fi
    else
        CUDA_AVAILABLE=$(conda run -n ml python -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')" 2>/dev/null || echo "NO")
        if [ "$CUDA_AVAILABLE" == "YES" ]; then
            CURRENT_CUDA=$(conda run -n ml python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
            echo "CUDA is available (version $CURRENT_CUDA)"
        else
            echo "WARNING: CUDA not available. GPU acceleration will not work"
        fi
    fi
fi

# Check if transformers is installed
TRANSFORMERS_INSTALLED=$(conda run -n ml python -c "import transformers; print('YES')" 2>/dev/null || echo "NO")
if [ "$TRANSFORMERS_INSTALLED" != "YES" ]; then
    echo "Step 6: Installing HuggingFace packages..."
    conda run -n ml pip install transformers datasets accelerate
    echo "HuggingFace packages installed."
else
    echo "Step 6: HuggingFace packages already installed, skipping."
fi

# If requirements.txt exists, install pure Python packages with pip
if [ -f "requirements.txt" ]; then
    echo "Step 7: Installing additional packages from requirements.txt..."
    conda run -n ml pip install -r requirements.txt --no-cache-dir
    echo "Additional packages installed."
else
    echo "Step 7: No requirements.txt found. Skipping additional package installation."
fi

# Check if ipykernel is installed
IPYKERNEL_INSTALLED=$(conda run -n ml python -c "import ipykernel; print('YES')" 2>/dev/null || echo "NO")
if [ "$IPYKERNEL_INSTALLED" != "YES" ]; then
    echo "Step 8: Configuring Jupyter support..."
    conda install -n ml -y ipykernel
    conda run -n ml python -m ipykernel install --user --name ml --display-name "Python (ML)"
    echo "Jupyter configuration complete."
else
    echo "Step 8: Jupyter already configured, skipping."
fi

# Verify installation
echo "Step 9: Verifying PyTorch installation..."
if [[ "$PLATFORM" == "macos-arm64" ]]; then
    conda run -n ml python -c "import torch; print('PyTorch version:', torch.__version__); print('MPS available:', torch.backends.mps.is_available()); print('Device:', 'mps' if torch.backends.mps.is_available() else 'cpu')"
    
else
    conda run -n ml python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
fi

# Step 10: Install and configure rclone for experiment syncing
echo "Step 10: Setting up rclone for experiment syncing..."
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    
    # Check if unzip is available (required for rclone installation)
    if ! command -v unzip &> /dev/null; then
        echo "Installing unzip (required for rclone)..."
        if command -v apt-get &> /dev/null; then
            apt-get update -qq && apt-get install -y unzip
        elif command -v yum &> /dev/null; then
            yum install -y unzip
        elif command -v brew &> /dev/null; then
            brew install unzip
        else
            echo "Warning: Could not install unzip automatically. Please install it manually."
        fi
    fi
    
    curl https://rclone.org/install.sh | bash
    echo "rclone installed successfully."
else
    echo "rclone already installed, checking version..."
    rclone version
fi

# Check if rclone is configured
if [ ! -f "$HOME/.config/rclone/rclone.conf" ] || [ ! -s "$HOME/.config/rclone/rclone.conf" ]; then
    echo ""
    echo "=========================================="
    echo "RCLONE CONFIGURATION REQUIRED"
    echo "=========================================="
    echo "rclone is installed but not configured."
    echo "To sync your experiments directory, you need to configure a cloud storage backend."
    echo ""
    echo "Run this command to configure rclone:"
    echo "  rclone config"
    echo ""
    echo "Recommended backends for simplicity:"
    echo "  - Google Drive (15GB free)"
    echo "  - Dropbox (2GB free)"
    echo "  - OneDrive (5GB free)"
    echo ""
    echo "After configuration, you can sync experiments with:"
    echo "  ./sync_experiments.sh push    # Upload experiments to cloud"
    echo "  ./sync_experiments.sh pull    # Download experiments from cloud"
    echo "=========================================="
else
    echo "rclone already configured. Available remotes:"
    rclone listremotes
fi

echo "=== Setup complete! ==="
echo "Activate environment with: conda activate ml"