#!/bin/bash
# Simple setup for ReAct Python Bug-Fixing Agent

ENV_NAME="agent_env"

echo "Creating Conda environment from environment.yml..."
source $(conda info --base)/etc/profile.d/conda.sh
conda env create -f environment.yml -n $ENV_NAME

echo "Activating environment..."
conda activate $ENV_NAME

# Create logs folder
mkdir -p logs

echo "Environment '$ENV_NAME' is ready! All runs will be logged to logs/agent_runs.log"
