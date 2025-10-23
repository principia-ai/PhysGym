#!/bin/bash

mode="default"
log_file="logs/baseline_${mode}_gpt_oss.log"

# Change to project root directory
cd "$(dirname "$0")/.."

python experiments/run_baseline.py \
    --mode $mode \
    --llm-model gpt-oss:20b \
    --api-provider ollama \
    --max-iterations 20 \
    --sample-quota 100 | tee -a $log_file