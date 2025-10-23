#!/bin/bash
# This script is used to run the baseline model for the given dataset.

LLM="google/gemini-2.5-flash" # "deepseek/deepseek-chat-v3.1" # "google/gemini-2.5-flash" # "google/gemini-2.5-flash-preview:thinking"
mode="default" # "default", "no_context", "no_description", "no_description_anonymous"
log_file="logs/test_${mode}_test.log"

# Change to project root directory
cd "$(dirname "$0")/.."

python experiments/run_baseline.py \
    --llm-model $LLM \
    --env-id 285 \
    --api-provider "openrouter" \
    --mode $mode