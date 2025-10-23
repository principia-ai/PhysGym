#!/usr/bin/env python3
"""
Run baseline LLM researcher experiments.

This script runs physics equation discovery experiments using the baseline
LLM researcher with various models and configurations.
"""

import datetime
from dataclasses import dataclass
import json
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import methods module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from installed physgym package
from physgym.interface import ResearchInterface, ExperimentRunState, setup_logging
from methods.baseline_researcher import BaselineResearcher
from physgym.utils.llm_providers import get_recommended_provider, show_provider_status, load_api_key


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run baseline experiment with full samples')
    parser.add_argument('--env-id', type=int, default=None,
                        help='Environment ID to run (if not specified, run range of environments from full_samples.json)')
    parser.add_argument('--mode', type=str, default="default", help='Experiment mode')
    parser.add_argument('--llm-model', type=str,
                        default='google/gemini-2.5-flash',
                        help='LLM model to use')
    parser.add_argument('--api-key-file', type=str,
                        default='api_keys.env', help='Path to the API key file')
    parser.add_argument('--api-provider', type=str,
                        default=None, 
                        help='LLM provider (openrouter, openai, anthropic, deepseek, ollama, vllm)')
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Maximum number of iterations (overrides config)')
    parser.add_argument('--sample-quota', type=int, default=None,
                        help='Sample quota (overrides config)')
    parser.add_argument('--experiments-per-iteration', type=int, default=None,
                        help='Experiments per iteration (overrides config)')
    parser.add_argument('--base-url', type=str, default=None,
                        help='Base URL for local LLM servers (e.g., http://localhost:8000)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--idx-start', type=int, default=0,
                        help='Starting index for experiment range (default: 0)')
    parser.add_argument('--idx-end', type=int, default=None,
                        help='Ending index for experiment range (default: all experiments)')
    return parser.parse_args()

@dataclass
class ExperimentConfig:
    """Configuration for the experiment run."""
    env_id: int = 411 # 41 54 134 285 674
    experiments_per_iteration: int = 20
    max_iterations: int = 20
    sample_quota: int = 100
    initial_test_quota: int = 2
    mode: str = "no_context" # "default", "no_context", "no_description"
    api_key_file: str = "api_keys.env"
    llm_model: str = "google/gemini-2.5-flash-preview:thinking" # "google/gemini-2.5-flash-preview:thinking", "deepseek/deepseek-r1", "deepseek/deepseek-r1:free", "google/gemini-2.5-flash-preview", "openai/gpt-4.1-mini", "openai/o4-mini-high", 
    api_provider: str = "openrouter"  # LLM provider
    base_url: str = None  # Base URL for local LLM servers (optional)
    capture_thinking: bool = True
    verbose: bool = False
    history_folder_template: str = "../histories/baseline/{mode}/{llm_model_slug}"  # Relative to experiments/

    def get_history_folder(self) -> str:
        """Generates the history folder path based on the LLM model and provider."""
        llm_model_slug = self.llm_model.replace("/", "_").replace(":", "_")
        if self.llm_model == "deepseek-r1-250120":
            llm_model_slug = "deepseek_deepseek-r1"
        
        # Add provider prefix for local LLMs
        if self.api_provider in ["ollama", "vllm"]:
            llm_model_slug = f"{self.api_provider}_{llm_model_slug}"
            
        return self.history_folder_template.format(mode=self.mode, llm_model_slug=llm_model_slug)

def load_sample_ids(sample_file):
    """Load the environment IDs from the samples file."""
    try:
        # First try the provided path relative to project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        sample_path = project_root / sample_file
        
        if not sample_path.exists():
            # Try within physgym package
            import physgym
            package_dir = Path(physgym.__file__).parent
            sample_path = package_dir / "samples" / "full_samples.json"
        
        with open(sample_path, 'r') as f:
            samples = json.load(f)
        return [sample['id'] for sample in samples]
    except Exception as e:
        print(f"Error loading samples: {e}")
        sys.exit(1)

def process_researcher_output(result: dict, exp: ResearchInterface, run_state: ExperimentRunState, logger: callable):
    """Processes and logs the output from the researcher."""
    current_hypothesis = result.get('current_hypothesis_formula', 'None')
    test_flag = result.get('test_hypothesis_flag', False)
    proposed_experiments = result.get('next_experiments', [])

    analysis_log = {
        "iteration": run_state.current_iteration,
        "LLM_input": result.get("input"),
        "observations_count": len(exp.get_observations()) + len(exp.get_hypothesis_history()),
        "current_hypothesis": current_hypothesis,
        "test_hypothesis_flag": test_flag,
        "proposed_experiments_count": len(proposed_experiments),
        "remaining_sample_quota": exp.get_remaining_quota(),
        "remaining_test_quota": exp.test_quota,
        "model_thinking": result.get("thinking"), # Include thinking if available
        "raw_response": result.get("raw_response") # Include raw response if available
    }
    logger("researcher_analysis", analysis_log)

    print("\nResearcher Results:")
    print(f"  Hypothesis: {current_hypothesis}")
    print(f"  Test Hypothesis Flag: {test_flag}")
    print(f"  Proposed Experiments ({len(proposed_experiments)}):")
    for i, exp_params in enumerate(proposed_experiments, 1):
        print(f"    Experiment {i}: {exp_params}")

    return current_hypothesis, test_flag, proposed_experiments


def run_experiment_iterations(
    config: ExperimentConfig,
    exp: ResearchInterface,
    researcher: BaselineResearcher,
    run_state: ExperimentRunState,
    logger: callable
):
    """Main loop for running experiment iterations."""

    # Prepare the inputs for the researcher
    problem_desc = exp.problem_content
    controllable_vars = exp.controllable_variables
    observable_var = exp.observable_var

    while exp.get_remaining_quota() > 0 and exp.test_quota > 0 and run_state.current_iteration < config.max_iterations:
        run_state.current_iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {run_state.current_iteration}/{config.max_iterations} - "
              f"Sample Quota: {exp.get_remaining_quota()}/{config.sample_quota} - "
              f"Test Quota: {exp.test_quota}/{config.initial_test_quota}")
        print(f"{'='*60}")

        historical_data = exp.get_observations() + exp.get_hypothesis_history()
        print(f"Total historical data points for researcher: {len(historical_data)}")

        print("\nRunning Baseline LLM Researcher to design experiments...")
        researcher_result = researcher.analyze_and_propose(
            problem_description=problem_desc,
            controllable_variables=controllable_vars,
            observable_variable=observable_var,
            historical_experiments=historical_data,
            quota={"experiments_quota": min(exp.get_remaining_quota(), config.experiments_per_iteration), # Propose for available iteration budget
                   "test_quota": exp.test_quota},
            capture_thinking=config.capture_thinking
        )

        current_hypothesis, test_flag, proposed_experiments = process_researcher_output(
            researcher_result, exp, run_state, logger
        )

        # Select experiments to run based on available quota for this iteration
        experiments_to_run_now = proposed_experiments[:config.experiments_per_iteration]

        if not experiments_to_run_now:
            if exp.get_remaining_quota() > 0 : # Only break if no proposals AND quota left (otherwise loop terminates naturally)
                print("\nNo experiments proposed by the researcher.")
                logger("no_experiments_proposed", {"iteration": run_state.current_iteration})
            else: # Out of quota, let the main loop condition handle this.
                 print("\nNo experiments proposed, and sample quota likely exhausted.")


        # Use the execute_experiments_and_evaluate method from the Experiment class
        exp.execute_experiments_and_evaluate(
            experiments_to_run_now, 
            current_hypothesis,
            test_flag, run_state, logger
        )

        if run_state.best_hypothesis_info["is_correct"]:
            print("\nCorrect hypothesis found and verified. Ending iterations early.")
            break

        print(f"\nEnd of Iteration {run_state.current_iteration}. Remaining sample quota: {exp.get_remaining_quota()}. "
              f"Remaining test quota: {exp.test_quota}.")

    if exp.get_remaining_quota() <= 0:
        print("\nSample quota exhausted.")
    if run_state.current_iteration >= config.max_iterations:
        print("\nMaximum iterations reached.")


def main(config: ExperimentConfig = None):
    """Main function to orchestrate the experiment."""
    script_start_time = datetime.datetime.now()
    print(f"Starting experiment run at {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if config.verbose:
        print(f"Configuration:")
        print(f"  Provider: {config.api_provider}")
        print(f"  Model: {config.llm_model}")
        print(f"  Base URL: {config.base_url}")
        print(f"  Environment ID: {config.env_id}")
        print(f"  Mode: {config.mode}")
    
    # Handle API key loading (may not be needed for local LLMs)
    api_key = None
    if config.api_provider in ["ollama", "vllm"]:
        if config.verbose:
            print("Using local LLM - no API key required")
    else:
        api_key = load_api_key(env_file=config.api_key_file, provider=config.api_provider)
        if not api_key:
            print(f"Error: API key not found for provider {config.api_provider}")
            print(f"Please check your {config.api_key_file} file")
            return

    # Show provider status if verbose mode
    if config.verbose:
        print("\nLLM Provider Status:")
        show_provider_status()

    # Setup logging using the centralized setup_logging function
    history_folder = config.get_history_folder()
    experiment_log = {
        "sample_quota": config.sample_quota,
        "test_quota": config.initial_test_quota,
        "max_iterations": config.max_iterations,
        "experiments_per_iteration": config.experiments_per_iteration,
        "provider": config.api_provider,
        "base_url": config.base_url
    }
    log_file_path, logger_func = setup_logging(config.env_id, config.llm_model, history_folder, experiment_log)

    # Initialize the experiment run state
    run_state = ExperimentRunState(log_file_path=log_file_path)

    # Create the experiment instance
    history_file = f"{history_folder}/experiment_{config.env_id}_history.json"
    exp = ResearchInterface(config.env_id, config.sample_quota, config.initial_test_quota, config.mode, history_file=history_file)
    print(f"Experiment initialized: {exp}")

    # Initialize the baseline researcher
    researcher = BaselineResearcher(
        api_key=api_key,
        model=config.llm_model,
        env_file=config.api_key_file,
        provider=config.api_provider
    )
    print(f"BaselineResearcher initialized - Provider: {config.api_provider}, Model: {config.llm_model}")

    # Run the experiment iterations
    run_experiment_iterations(config, exp, researcher, run_state, logger_func)

    # Generate the final report using the method from ResearchInterface class
    results = exp.generate_final_report(run_state, logger_func)

    script_end_time = datetime.datetime.now()
    print(f"Research run finished at {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {script_end_time - script_start_time}")

    # Return results for potential external use
    return results


if __name__ == "__main__":
    args = parse_args()
    
    # Auto-detect provider if using default
    if args.api_provider is None:
        recommended = get_recommended_provider()
        args.api_provider = recommended
        print(f"Auto-detected LLM provider: {recommended}")
    
    # Try local path first, will fallback to package path if not found
    sample_file = 'physgym/samples/full_samples.json'
    env_ids = load_sample_ids(sample_file)

    if args.env_id is not None:
        if args.env_id not in env_ids:
            print(f"Error: Environment ID {args.env_id} not found in {sample_file}")
            sys.exit(1)
        print(f"Running experiment for environment ID: {args.env_id}")
        
        # Create config with all arguments
        config = ExperimentConfig(
            env_id=args.env_id, 
            mode=args.mode, 
            llm_model=args.llm_model, 
            api_key_file=args.api_key_file,
            api_provider=args.api_provider,
            base_url=args.base_url,
            verbose=args.verbose,
            max_iterations=args.max_iterations or 20,
            sample_quota=args.sample_quota or 100,
            experiments_per_iteration=args.experiments_per_iteration or 20
        )
        
        results = main(config=config)
        print(f"Results for environment ID {args.env_id}: {results}")
    else:
        print(f"Running experiments for environment IDs in {sample_file}")
        # Determine the index range
        total_envs = len(env_ids)
        start_idx = args.idx_start
        end_idx = args.idx_end if args.idx_end is not None else total_envs            
        print(f"Running experiments for indices {start_idx} to {end_idx-1} (total: {end_idx-start_idx} environments)")
        idx_range = range(start_idx, end_idx)
        for idx in idx_range:
            env_id = env_ids[idx]
            print(f"\nRunning experiment for environment ID: {env_id}")
            
            # Create config with all arguments
            config = ExperimentConfig(
                env_id=env_id, 
                mode=args.mode, 
                llm_model=args.llm_model, 
                api_key_file=args.api_key_file,
                api_provider=args.api_provider,
                base_url=args.base_url,
                verbose=args.verbose,
                max_iterations=args.max_iterations or 20,
                sample_quota=args.sample_quota or 100,
                experiments_per_iteration=args.experiments_per_iteration or 20
            )
            
            results_file = f"{config.get_history_folder()}/experiment_{env_id}_results.json"
            # Check if results file already exists
            if os.path.exists(results_file):
                print(f"Results for environment ID {env_id} already exist. Skipping...")
                continue
                
            results = main(config=config)
            
            # save results to a file
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            