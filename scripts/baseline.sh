#!/bin/bash
# This script is used to run the baseline model for the given dataset.

LLM="google/gemini-2.5-pro" # "google/gemini-2.5-flash"
mode="default"
log_file="logs/baseline_${mode}_gemini.log"

# Change to project root directory
cd "$(dirname "$0")/.."

python experiments/run_baseline.py \
    --llm-model $LLM \
    --api-provider "openrouter" \
    --idx-start 0 \
    --idx-end 97 \
    --mode $mode | tee -a $log_file