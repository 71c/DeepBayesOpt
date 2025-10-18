#!/bin/bash

# Check if the specific conda installation exists
if [ -f "/share/apps/anaconda3/2022.10/bin/conda" ]; then
    eval "$(/share/apps/anaconda3/2022.10/bin/conda shell.bash hook)"
    conda activate alon2
else
    echo "Warning: Conda installation not found at /share/apps/anaconda3/2022.10/bin/conda"
    echo "Please ensure conda is installed and available in your PATH, or update the path in this script."
    exit 1
fi
