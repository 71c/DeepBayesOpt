#!/bin/bash
# Start an interactive SLURM session with GPU and activate the environment

srun -p frazier-interactive -t 16:00:00 --mem=64gb --gres=gpu:1 --pty \
    bash --rcfile ~/projects/DeepBayesOpt/.session_init.sh
