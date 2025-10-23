# PhysGym Installation Guide

This guide explains how to install and use PhysGym as a proper Python package.

## Installation

### Development Installation (Editable Mode)

For development work or if you've cloned the repository:

```bash
# Navigate to the project directory
cd /path/to/Physicist

# Install in editable mode (changes to the code take effect immediately)
pip install -e .
```

### Installation with Optional Dependencies

PhysGym has optional dependencies for different use cases:

```bash
# Install with local LLM support (torch, ollama, vllm)
pip install -e ".[local-llm]"

# Install with development tools (pytest, black, ruff)
pip install -e ".[dev]"

# Install everything
pip install -e ".[all]"
```

## Verification

After installation, you can verify that PhysGym is installed correctly:

```python
import physgym
print(f"PhysGym version: {physgym.__version__}")

# Test loading an environment
env = physgym.PhyEnv(285)
print(f"Loaded environment: {env.id}")

# Test creating a research interface
experiment = physgym.ResearchInterface(285, sample_quota=50)
print(f"Experiment ready with {experiment.get_remaining_quota()} samples")
```

## Usage

Once installed, you can import and use PhysGym from anywhere on your system:

```python
import physgym
import numpy as np

# Create an experiment with a specific physics problem ID
experiment = physgym.ResearchInterface(285, sample_quota=50)

# Generate random input samples
input_samples = []
for _ in range(5):
    sample = {}
    for param in experiment.all_params:
        sample[param] = np.random.uniform(0.1, 10.0)
    input_samples.append(sample)

# Run the experiment
results = experiment.run_experiment(input_samples)
print(results)
```

## Available Exports

PhysGym exports the following main components:

### Core Classes
- `PhyEnv`: Physics environment class for loading and running physics simulations
- `ResearchInterface`: Main interface for running physics discovery experiments
- `ExperimentRunState`: State tracking for experiment runs
- `setup_logging`: Function to set up experiment logging

### Utility Functions
- `evaluate_hypothesis`: Evaluate a hypothesis function against true physics
- `create_function_from_string`: Safely create executable functions from strings
- `get_recommended_provider`: Auto-detect available LLM providers
- `show_provider_status`: Display status of all LLM providers
- `load_api_key`: Load API keys from environment files

## Package Data

PhysGym includes the following data files within the packages:

- **Samples**: `physgym/samples/full_samples.json` - Pre-processed physics problems
- **PhysGym Prompts**: `physgym/prompts/*.txt` - LLM prompt templates for preprocessing and evaluation
- **Method Prompts**: `methods/prompts/*.txt` - Researcher prompt templates for baseline and workflow methods

These files are automatically included when you install the package and are accessible via the package's internal path resolution.

## Running Experiments

After installation, you can run experiments using the included scripts:

```bash
# Run a baseline experiment with a specific environment
python baseline_experiment.py --env-id 285 --llm-model google/gemini-2.5-flash

# Run experiments for all samples in the dataset
python baseline_experiment.py --idx-start 0 --idx-end 10
```

## Updating the Package

Since the package is installed in editable mode (`-e`), any changes you make to the source code will automatically take effect without needing to reinstall. However, if you:

- Change `pyproject.toml` configuration
- Add new package data files
- Modify package structure

You should reinstall the package:

```bash
pip install -e . --force-reinstall --no-deps
```

## Uninstalling

To uninstall PhysGym:

```bash
pip uninstall physgym
```

## Publishing to PyPI (Future)

When ready to publish to PyPI:

```bash
# Build the distribution
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

Then users can install with:

```bash
pip install physgym
```

