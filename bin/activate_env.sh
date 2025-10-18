#!/bin/bash

# Check if the specific conda installation exists
if [ -f "/share/apps/anaconda3/2022.10/bin/conda" ]; then
    eval "$(/share/apps/anaconda3/2022.10/bin/conda shell.bash hook)"
    conda activate nn_bo
else
    # Try to use conda from PATH
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate nn_bo
    else
        exit 1
    fi
fi
