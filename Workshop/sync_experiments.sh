#!/bin/bash
# Sync script for experiments directory using rclone
# Usage: ./sync_experiments.sh [push|pull|status]
#
# Examples:
#   ./sync_experiments.sh push    # Upload experiments to cloud
#   ./sync_experiments.sh pull    # Download experiments from cloud  
#   ./sync_experiments.sh status  # Show sync status and file counts

set -e

# Configuration
EXPERIMENTS_DIR="experiments"
REMOTE_NAME=""  # Will be auto-detected or set manually
REMOTE_PATH="pollux-experiments"  # Remote directory name

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    log_error "rclone is not installed. Please run ./envsetup.sh first."
    exit 1
fi

# Auto-detect remote name (use first configured remote)
if [ -z "$REMOTE_NAME" ]; then
    REMOTE_NAME=$(rclone listremotes | head -n1 | sed 's/://')
    if [ -z "$REMOTE_NAME" ]; then
        log_error "No rclone remotes configured. Please run 'rclone config' first."
        exit 1
    fi
    log_info "Using remote: $REMOTE_NAME"
fi

# Create experiments directory if it doesn't exist
mkdir -p "$EXPERIMENTS_DIR"

case "${1:-status}" in
    "push"|"upload")
        log_info "Uploading experiments to $REMOTE_NAME:$REMOTE_PATH..."
        
        # Show what will be synced
        echo ""
        log_info "Files to upload:"
        rclone size "$EXPERIMENTS_DIR" --human-readable
        
        # Confirm before large uploads
        if [ -d "$EXPERIMENTS_DIR" ] && [ "$(du -s "$EXPERIMENTS_DIR" | cut -f1)" -gt 1000000 ]; then  # > 1GB
            echo ""
            read -p "This will upload a large amount of data. Continue? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Upload cancelled."
                exit 0
            fi
        fi
        
        # Perform the sync
        rclone sync "$EXPERIMENTS_DIR" "$REMOTE_NAME:$REMOTE_PATH" \
            --progress \
            --transfers=4 \
            --checkers=8 \
            --fast-list \
            --exclude="*.tmp" \
            --exclude="**/__pycache__/**" \
            --exclude="**/wandb/run-**/files/media/**"  # Skip large wandb media files
        
        log_success "Upload completed!"
        ;;
        
    "pull"|"download")
        log_info "Downloading experiments from $REMOTE_NAME:$REMOTE_PATH..."
        log_warning "Using safe copy mode - no local files will be deleted"
        
        # Check if remote directory exists
        if ! rclone lsd "$REMOTE_NAME:" | grep -q "$REMOTE_PATH"; then
            log_warning "Remote directory $REMOTE_NAME:$REMOTE_PATH not found."
            log_info "Available directories:"
            rclone lsd "$REMOTE_NAME:" || log_warning "No directories found on remote."
            exit 1
        fi
        
        # Show what will be downloaded
        echo ""
        log_info "Files to download:"
        rclone size "$REMOTE_NAME:$REMOTE_PATH" --human-readable
        
        # Perform the copy (safe - never deletes local files)
        rclone copy "$REMOTE_NAME:$REMOTE_PATH" "$EXPERIMENTS_DIR" \
            --progress \
            --transfers=4 \
            --checkers=8 \
            --fast-list \
            --update \
            --exclude="*.tmp" \
            --exclude="**/__pycache__/**"
        
        log_success "Download completed! (Local files preserved)"
        ;;
        
    "status")
        log_info "Sync status for experiments directory"
        echo ""
        
        # Local info
        if [ -d "$EXPERIMENTS_DIR" ]; then
            log_info "Local experiments directory:"
            du -sh "$EXPERIMENTS_DIR"
            find "$EXPERIMENTS_DIR" -type f | wc -l | xargs printf "  Files: %d\n"
            find "$EXPERIMENTS_DIR" -name "*.safetensors" | wc -l | xargs printf "  Model files (.safetensors): %d\n"
        else
            log_warning "Local experiments directory not found."
        fi
        
        echo ""
        
        # Remote info
        if rclone lsd "$REMOTE_NAME:" | grep -q "$REMOTE_PATH"; then
            log_info "Remote experiments directory ($REMOTE_NAME:$REMOTE_PATH):"
            rclone size "$REMOTE_NAME:$REMOTE_PATH" --human-readable
            rclone size "$REMOTE_NAME:$REMOTE_PATH" --json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'  Files: {data[\"count\"]}')
"
        else
            log_warning "Remote experiments directory not found."
        fi
        
        echo ""
        log_info "To sync experiments:"
        echo "  Upload:   ./sync_experiments.sh push"
        echo "  Download: ./sync_experiments.sh pull"
        ;;
        
    "help"|"-h"|"--help")
        echo "Experiment Sync Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  push     Upload local experiments to cloud storage"
        echo "  pull     Download experiments from cloud storage"
        echo "  status   Show local and remote sync status (default)"
        echo "  help     Show this help message"
        echo ""
        echo "Configuration:"
        echo "  Remote: $REMOTE_NAME"
        echo "  Path: $REMOTE_PATH"
        echo ""
        echo "To configure rclone: rclone config"
        ;;
        
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information."
        exit 1
        ;;
esac 