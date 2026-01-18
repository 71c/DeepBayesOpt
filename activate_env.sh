#!/bin/bash

export 'PYTORCH_ALLOC_CONF=expandable_segments:True'

# Accept env_name as first argument, default to nn_bo
env_name=${1:-nn_bo}


# Check if the specific conda installation exists
if [ -f "/share/apps/anaconda3/2022.10/bin/conda" ]; then
    eval "$(/share/apps/anaconda3/2022.10/bin/conda shell.bash hook)"
    conda activate $env_name
else
    # Try to use conda from PATH
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate $env_name
    else
        exit 1
    fi
fi

echo "Activated conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
