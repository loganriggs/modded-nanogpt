#!/bin/bash

# Script to run all architecture and size combinations sequentially
# Each experiment uses all 8 GPUs via DDP
# 8 architectures × 2 sizes = 16 total runs

set -e

# Define the 8 architecture configurations
# Format: "name|flags"
ARCHITECTURES=(
    "default|"
    "squared_attn|--squared_attn"
    "squared_mlp|--squared_mlp"
    "squared_mlp_attn|--squared_mlp --squared_attn"
    "bilinear|--bilinear"
    "bilinear_attn|--bilinear --squared_attn"
    "bilinear_gated|--bilinear --gated"
    "bilinear_gated_attn|--bilinear --gated --squared_attn"
)

# Create logs directory if it doesn't exist
mkdir -p logs

# Array to store all experiment configurations
# Format: "arch_name|flags|size_setting"
EXPERIMENTS=()

# Generate all experiment configurations
for i in "${!ARCHITECTURES[@]}"; do
    IFS='|' read -r arch_name arch_flags <<< "${ARCHITECTURES[$i]}"

    # Determine which size settings to use
    if [[ $arch_name == bilinear* ]]; then
        # Bilinear architectures use bilinear_small and bilinear_medium
        for size in "small" "medium"; do
            size_setting="bilinear_${size}"
            EXPERIMENTS+=("$arch_name|$arch_flags|$size_setting")
        done
    else
        # Non-bilinear architectures use small and medium
        for size_setting in "small" "medium"; do
            EXPERIMENTS+=("$arch_name|$arch_flags|$size_setting")
        done
    fi
done

echo "=========================================="
echo "Running All Experiments"
echo "=========================================="
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "Each experiment will use all 8 GPUs via DDP"
echo ""

# Check for upload flag and HF_USERNAME early
UPLOAD_MODELS=false
if [ "$1" == "--upload" ]; then
    UPLOAD_MODELS=true
    if [ -z "$HF_USERNAME" ]; then
        echo "ERROR: HF_USERNAME environment variable not set"
        echo "Set it with: export HF_USERNAME=your-username"
        exit 1
    fi
    echo "Will upload models to HuggingFace after each experiment completes"
    echo ""
fi

# Track experiment results
SUCCESSFUL=0
FAILED=0
FAILED_EXPERIMENTS=()
UPLOADED=0

# Run experiments sequentially
for ((i=0; i<${#EXPERIMENTS[@]}; i++)); do
    IFS='|' read -r arch_name arch_flags size_setting <<< "${EXPERIMENTS[$i]}"

    exp_name="${arch_name}_${size_setting}"
    log_file="logs/run_${exp_name}.log"

    echo ""
    echo "=========================================="
    echo "Experiment $((i+1))/${#EXPERIMENTS[@]}: $exp_name"
    echo "=========================================="
    echo "Architecture: $arch_name"
    echo "Size setting: $size_setting"
    echo "Flags: $arch_flags"
    echo "Log file: $log_file"
    echo ""
    echo "Starting at: $(date)"
    echo ""

    # Run the experiment
    if bash run.sh --setting "$size_setting" $arch_flags > "$log_file" 2>&1; then
        echo "✓ SUCCESS: $exp_name"
        ((SUCCESSFUL++))

        # Upload to HuggingFace if requested
        if [ "$UPLOAD_MODELS" = true ]; then
            echo ""
            echo "Uploading $exp_name to HuggingFace..."

            # Find the checkpoint file for this experiment
            checkpoint=$(find logs/gpt2_* -name "state_step*.pt" -newer "$log_file" 2>/dev/null | head -1)

            if [ -n "$checkpoint" ]; then
                # Extract run name from checkpoint path
                run_name=$(basename $(dirname "$checkpoint"))
                repo_id="${HF_USERNAME}/${run_name}"

                if python upload_to_hf.py --checkpoint "$checkpoint" --repo_id "$repo_id"; then
                    echo "✓ Uploaded: $run_name to https://huggingface.co/$repo_id"
                    ((UPLOADED++))
                else
                    echo "✗ Failed to upload: $run_name"
                fi
            else
                echo "⚠ Warning: No checkpoint found for $exp_name"
            fi
            echo ""
        fi
    else
        echo "✗ FAILED: $exp_name (exit code: $?)"
        ((FAILED++))
        FAILED_EXPERIMENTS+=("$exp_name")
    fi

    echo "Completed at: $(date)"
    echo "Progress: $((i+1))/${#EXPERIMENTS[@]} | Success: $SUCCESSFUL | Failed: $FAILED"
    if [ "$UPLOAD_MODELS" = true ]; then
        echo "Uploaded: $UPLOADED"
    fi
done

echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo "Total: ${#EXPERIMENTS[@]} experiments"
echo "Successful: $SUCCESSFUL"
echo "Failed: $FAILED"
if [ "$UPLOAD_MODELS" = true ]; then
    echo "Uploaded to HuggingFace: $UPLOADED"
fi
echo ""

if [ $FAILED -gt 0 ]; then
    echo "Failed experiments:"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - $exp"
    done
    echo ""
fi

# List all log files
echo "Log files created:"
ls -lh logs/run_*.log 2>/dev/null || echo "No log files found"
echo ""

echo ""
echo "Done!"
