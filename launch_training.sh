#!/bin/bash
# Launch script for CodeLlama fine-tuning with DeepSpeed
# Usage: ./launch_training.sh <csv_path> [output_dir]

set -e  # Exit on any error

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

# Check if CSV path is provided
if [ $# -lt 1 ]; then
    print_error "Usage: $0 <csv_path> [output_dir]"
    print_error "Example: $0 ./training_data.csv ./results"
    exit 1
fi

CSV_PATH="$1"
OUTPUT_DIR="${2:-./results}"

print_status "Starting CodeLlama Fine-tuning Setup"
echo "======================================"

# Check if CSV file exists
if [ ! -f "$CSV_PATH" ]; then
    print_error "CSV file not found: $CSV_PATH"
    exit 1
fi

print_success "CSV file found: $CSV_PATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_success "Output directory created: $OUTPUT_DIR"

# Check GPU availability
print_status "Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
print_success "Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 4 ]; then
    print_warning "Expected 4 GPUs, found $GPU_COUNT. Training will continue but may be slower."
fi

# Check if DeepSpeed config exists
if [ ! -f "./ds_config.json" ]; then
    print_error "DeepSpeed config file not found: ./ds_config.json"
    print_error "Please ensure ds_config.json is in the current directory"
    exit 1
fi

print_success "DeepSpeed config found"

# Check if training script exists
if [ ! -f "./train.py" ]; then
    print_error "Training script not found: ./train.py"
    exit 1
fi

print_success "Training script found"

# Backup previous results if they exist
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
    BACKUP_DIR="${OUTPUT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    print_warning "Output directory not empty. Creating backup: $BACKUP_DIR"
    mv "$OUTPUT_DIR" "$BACKUP_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Log system information
print_status "System Information:"
echo "  - GPU Count: $GPU_COUNT"
echo "  - CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "  - CSV Path: $CSV_PATH"
echo "  - Output Directory: $OUTPUT_DIR"
echo ""

# Ask for confirmation
echo "Training Configuration:"
echo "  - Model: codellama/CodeLlama-7b-Instruct-hf"
echo "  - Max Sequence Length: 2048"
echo "  - LoRA Rank: 64"
echo "  - DeepSpeed ZeRO Stage: 2"
echo "  - Estimated Time: 2-8 hours"
echo ""

read -p "Continue with training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Training cancelled by user"
    exit 0
fi

# Start training
print_status "Starting training..."
print_status "Logs will be saved to training.log and $OUTPUT_DIR/"

# Create a comprehensive launch command
LAUNCH_CMD="deepspeed --num_gpus=$GPU_COUNT train.py \
    --csv_path '$CSV_PATH' \
    --output_dir '$OUTPUT_DIR' \
    --deepspeed_config ./ds_config.json \
    --max_seq_length 2048 \
    --lora_r 64 \
    --lora_alpha 16 \
    --test_size 0.1 \
    --validation_size 0.1"

# Execute the training
print_status "Executing: $LAUNCH_CMD"
echo ""

# Run with error handling
if eval $LAUNCH_CMD; then
    print_success "Training completed successfully!"
    print_success "Model saved to: $OUTPUT_DIR"
    
    # Show final statistics
    if [ -f "$OUTPUT_DIR/training_metrics.json" ]; then
        print_status "Training metrics saved to: $OUTPUT_DIR/training_metrics.json"
    fi
    
    if [ -f "training.log" ]; then
        FINAL_LOSS=$(tail -20 training.log | grep -o "Train Loss = [0-9.]*" | tail -1 | cut -d'=' -f2 | tr -d ' ')
        if [ ! -z "$FINAL_LOSS" ]; then
            print_success "Final training loss: $FINAL_LOSS"
        fi
    fi
    
    echo ""
    print_status "Next steps:"
    echo "  1. Test your model with the generated code"
    echo "  2. Check TensorBoard logs: tensorboard --logdir $OUTPUT_DIR/runs"
    echo "  3. Use the model for inference"
    
else
    print_error "Training failed!"
    print_error "Check training.log and $OUTPUT_DIR/ for error details"
    
    # Show last few lines of log for quick debugging
    if [ -f "training.log" ]; then
        print_status "Last 10 lines of training.log:"
        tail -10 training.log
    fi
    
    exit 1
fi

print_success "Script completed successfully!"
