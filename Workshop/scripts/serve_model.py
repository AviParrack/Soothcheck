#!/usr/bin/env python
# This script starts the FastAPI inference server for serving language models.
# It's a convenience wrapper around the main server module.
#
# Example usage:
# python -m scripts.serve_model gemma-3-27b-it --port 8000 --prompt jake-prompt
# python -m scripts.serve_model my_experiment --host 0.0.0.0 --device cuda

import sys
from pathlib import Path

# Ensure project root is in path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.server.main import main

if __name__ == "__main__":
    main() 