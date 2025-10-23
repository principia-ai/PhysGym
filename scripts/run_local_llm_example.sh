#!/bin/bash
# Example script showing how to run PhysGym experiments with local LLMs

# This script demonstrates different ways to run experiments/run_baseline.py with local LLMs

echo "🚀 PhysGym Local LLM Examples"
echo "=" * 50

# Change to project root directory
cd "$(dirname "$0")/.."

# Example 1: Using Ollama (if available)
echo "Example 1: Running with Ollama"
echo "Requirements: ollama serve, ollama pull llama3.2"
echo "Command:"
echo "python experiments/run_baseline.py \\"
echo "    --env-id 285 \\"
echo "    --mode default \\"
echo "    --llm-model llama3.2 \\"
echo "    --api-provider ollama \\"
echo "    --max-iterations 5 \\"
echo "    --sample-quota 20 \\"
echo "    --verbose"
echo ""

# Example 2: Using vLLM (if available)
echo "Example 2: Running with vLLM"
echo "Requirements: vLLM server running on localhost:8000"
echo "Command:"
echo "python experiments/run_baseline.py \\"
echo "    --env-id 285 \\"
echo "    --mode default \\"
echo "    --llm-model meta-llama/Llama-3.2-3B-Instruct \\"
echo "    --api-provider vllm \\"
echo "    --base-url http://localhost:8000 \\"
echo "    --max-iterations 5 \\"
echo "    --sample-quota 20 \\"
echo "    --verbose"
echo ""

# Example 3: Auto-detection of local LLM provider
echo "Example 3: Auto-detect local LLM provider (checks Ollama, then vLLM)"
echo "Command:"
echo "python experiments/run_baseline.py \\"
echo "    --env-id 285 \\"
echo "    --max-iterations 3 \\"
echo "    --verbose"
echo ""

# SLURM example
echo "Example 4: SLURM cluster with vLLM"
echo "Environment variables:"
echo "export LOCAL_MODEL_PATH=/path/to/your/model"
echo "export ENV_ID=285"
echo "export MODE=default"
echo "export MAX_ITERATIONS=10"
echo "export SAMPLE_QUOTA=50"
echo ""
echo "Command:"
echo "sbatch scripts/slurm.sh"
echo ""

echo "💡 Tips:"
echo "1. Check provider status: python -c 'from physgym.utils.llm_providers import show_provider_status; show_provider_status()'"
echo "2. Test local setup: python local_llm_examples.py"
echo "3. For development, use smaller quotas (--max-iterations 3 --sample-quota 10)"
echo "4. Use --verbose to see detailed progress"
echo ""

# Uncomment to actually run a test (requires local LLM setup)
# echo "Running a quick test..."
# python experiments/run_baseline.py --env-id 285 --api-provider ollama --max-iterations 1 --sample-quota 5 --verbose