#!/usr/bin/env python3
"""
Post-process experiment results and generate statistics.

This script analyzes experiment results from the histories directory,
computes metrics, and generates summary statistics.
"""

import os
import json
import argparse
import glob
import shutil
from collections import defaultdict
import statistics
from pathlib import Path
import sys

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from physgym.phyenv import PhyEnv

def extract_model_name(path):
    """Extract model name from the path."""
    parts = path.split('/')
    for i, part in enumerate(parts):
        if part == "histories":
            # Return the combination of experiment configuration and model
            if i + 2 < len(parts) and parts[i+2].startswith(('google', 'openai', 'deepseek', 'claude', 'vllm', 'ollama')):
                return f"{parts[i+1]}/{parts[i+2]}"
            elif i + 3 < len(parts) and parts[i+3].startswith(('google', 'openai', 'deepseek', 'claude', 'vllm', 'ollama')):
                return f"{parts[i+1]}/{parts[i+2]}/{parts[i+3]}"
    return "unknown"

def find_and_delete_unsuccessful_files(histories_dir, dry_run=True):
    """
    Find experiments where is_correct is False but still have remaining quotas,
    and delete all related files.
    
    Args:
        histories_dir (str): Directory containing history files
        dry_run (bool): If True, only print actions without deleting files
    
    Returns:
        int: Number of experiments deleted
    """
    log_files = glob.glob(f"{histories_dir}/**/*_log.jsonl", recursive=True)
    deleted_count = 0
    
    for log_file in log_files:
        delete_files = False
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get("event_type") == "experiment_complete":
                            is_correct = event["data"].get("best_hypothesis_is_correct", False)
                            sample_quota = event["data"].get("final_sample_quota_remaining", 0)
                            test_quota = event["data"].get("final_test_quota_remaining", 0)
                            
                            if not is_correct and sample_quota > 0 and test_quota > 0:
                                delete_files = True
                                experiment_id = log_file.split('_')[-3]  # Extract experiment ID
                                print(f"Found unsuccessful experiment {experiment_id} with remaining quotas:")
                                print(f"  Sample quota remaining: {sample_quota}")
                                print(f"  Test quota remaining: {test_quota}")
                                break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            continue
        
        if delete_files:
            # Get base path and experiment ID to delete related files
            base_path = os.path.dirname(log_file)
            experiment_id = os.path.basename(log_file).split('_')[1]  # Extract experiment ID
            
            # Find all related files for this experiment
            related_files = glob.glob(f"{base_path}/*_{experiment_id}_*")
            
            print(f"Found {len(related_files)} files to delete for experiment {experiment_id}:")
            for file in related_files:
                print(f"  - {file}")
            
            if not dry_run:
                for file in related_files:
                    try:
                        if os.path.isfile(file):
                            os.remove(file)
                        elif os.path.isdir(file):
                            shutil.rmtree(file)
                        print(f"Deleted: {file}")
                    except Exception as e:
                        print(f"Error deleting {file}: {e}")
            else:
                print("Dry run - no files deleted")
            
            deleted_count += 1
    
    return deleted_count
    
def process_log_file(log_file):
    """Process a single log file and extract metrics."""
    metrics = {
        "iterations": 0,
        "correct_found": False,
        "samples_used": 0,
        "samples_quota": 0,
        "test_used": 0,
        "test_quota": 0,
        "hypotheses_count": 0,
        "unique_hypotheses": set(),
        "success": False,
        "fit_quality": 0,
        "R2": 0
    }
    
    try:
        with open(log_file, 'r') as f:
            hypothesis_test_count = 0
            iter_count = 0
            is_best_so_far = False
            for line in f:
                event = json.loads(line.strip())
                event_type = event.get("event_type", "")
                
                # Extract initial quota information
                if event_type == "experiment_setup":
                    env_id = event["data"].get("env_id", "")
                    metrics["samples_quota"] = event["data"].get("sample_quota", 0)
                    metrics["test_quota"] = event["data"].get("test_quota", 0)
                    metrics["id"] = env_id
                
                # Count iterations and update samples used
                elif event_type == "researcher_analysis":
                    if hypothesis_test_count > 1:
                        break
                    iter_count += 1
                    iteration = event["data"].get("iteration", 0)
                    metrics["iterations"] = max(metrics["iterations"], iteration, iter_count)
                    metrics["samples_used"] = metrics["samples_quota"] - event["data"]["remaining_sample_quota"]
                    # Extract hypothesis information
                    hypothesis = event["data"].get("current_hypothesis", "")
                    if hypothesis:
                        metrics["hypotheses_count"] += 1
                        metrics["unique_hypotheses"].add(hypothesis)
                
                # Check if correct hypothesis was found
                elif event_type == "correct_hypothesis_found":
                    metrics["correct_found"] = True
                    metrics["success"] = True
                
                # Extract final metrics
                elif event_type == "experiment_complete":
                    break
                
                elif event_type == "hypothesis_testing":
                    hypothesis_test_count += 1
                    is_best_so_far = event["data"]["is_best_so_far"]
                    if is_best_so_far:
                        metrics["fit_quality"] = event["data"]["evaluation"]["fit_quality"]
                        metrics["R2"] = event["data"]["evaluation"]["fit_metrics"]["r2"]
                
                elif event_type == "experiment_execution":
                    if hypothesis_test_count > 1:
                        break
                        
                metrics["test_used"] = hypothesis_test_count

            # Try local path first, will fallback to package path if not found
            samples_file = 'physgym/samples/full_samples.json'
            if not os.path.exists(samples_file):
                from pathlib import Path
                import physgym
                package_dir = Path(physgym.__file__).parent
                samples_file = str(package_dir / "samples" / "full_samples.json")
            env = PhyEnv(env_id, samples_file)
            # Get difficulty factors
            metrics['answer_length'] = len(env.answer)
            metrics['num_input_var'] = len(env.input_variables_des)
            metrics['num_dummy_var'] = len(env.dummy_variables_des)
            metrics['solution_length'] = len(env.solution)
            if metrics['id'] is None:
                print(f"Error: No env_id found in {log_file}")
                exit()
        
        return metrics
    except Exception as e:
        print(f"Error processing {log_file}: {e} at Line {line}")
        return metrics

def main(histories_dir="histories"):
    """Main function to process experiment logs."""
    # Resolve path relative to this script (in analysis/)
    # If path is relative, make it relative to project root
    if not os.path.isabs(histories_dir):
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        histories_path = str((project_root / histories_dir).resolve())
    else:
        histories_path = histories_dir
    
    # Find all log files
    log_files = glob.glob(f"{histories_path}/*_log.jsonl", recursive=True)
    
    # Group by model
    models_data = defaultdict(list)
    
    for log_file in log_files:
        model = extract_model_name(log_file)
        metrics = process_log_file(log_file)
        models_data[model].append(metrics)
    
    # Compute aggregate statistics per model
    results = {}
    
    for model, experiments in models_data.items():
        if not experiments:
            continue
            
        success_rate = sum(1 for e in experiments if e["success"]) / len(experiments)
        avg_samples = statistics.mean(e["samples_used"] for e in experiments)
        avg_tests = statistics.mean(e["test_used"] for e in experiments)
        avg_iterations = statistics.mean(e["iterations"] for e in experiments)
        avg_hypotheses = statistics.mean(e["hypotheses_count"] for e in experiments)
        avg_unique_hypotheses = statistics.mean(len(e["unique_hypotheses"]) for e in experiments)
        
        results[model] = {
            "experiment_count": len(experiments),
            "success_rate": success_rate,
            "avg_samples_used": avg_samples,
            "avg_test_quota_used": avg_tests,
            "avg_iterations": avg_iterations,
            "avg_hypotheses_proposed": avg_hypotheses,
            "avg_unique_hypotheses": avg_unique_hypotheses,
        }
    
    # Print results in a readable format
    print("\nExperiment Results Summary by Model\n" + "="*40)
    
    for model, stats in results.items():
        print(f"\nModel: {model}")
        print(f"  Experiments: {stats['experiment_count']}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Avg Samples Used: {stats['avg_samples_used']:.2f}")
        print(f"  Avg Test Quota Used: {stats['avg_test_quota_used']:.2f}")
        print(f"  Avg Iterations: {stats['avg_iterations']:.2f}")
        print(f"  Avg Hypotheses Proposed: {stats['avg_hypotheses_proposed']:.2f}")
        print(f"  Avg Unique Hypotheses: {stats['avg_unique_hypotheses']:.2f}")
    
    # Save results to JSON
    output_file = os.path.join(histories_path, 'results_summary.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save raw models_data to JSON
    models_data_file = os.path.join(histories_path, 'models_data.json')
    # Convert sets to lists for JSON serialization
    serializable_data = {}
    for model, experiments in models_data.items():
        serializable_data[model] = []
        for exp in experiments:
            exp_copy = exp.copy()
            exp_copy['unique_hypotheses'] = list(exp_copy['unique_hypotheses'])
            serializable_data[model].append(exp_copy)
    
    with open(models_data_file, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"\nDetailed results saved to results_summary.json")
    print(f"Raw data saved to models_data.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process experiment histories.")
    parser.add_argument("--clean", action="store_true", help="Delete unsuccessful experiments")
    arg = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    baseline_dir = project_root / "histories" / "baseline"
    
    for mode in os.listdir(baseline_dir):
        mode_folder = baseline_dir / mode
        if not mode_folder.is_dir():
            continue
        for folder in os.listdir(mode_folder):
            if 'Qwen' in folder:
                histories_dir = mode_folder / folder
                print(f"Processing folder: {histories_dir}")
                if arg.clean:
                    deleted = find_and_delete_unsuccessful_files(str(histories_dir), dry_run=False)
                    print(f"Deleted {deleted} unsuccessful experiments in {histories_dir}")
                else:
                    main(str(histories_dir))
    
    # To delete for a specific model configuration:
    # delete_unsuccessful_experiments("histories/baseline/default/google_gemini-2.5-flash-preview:thinking", dry_run=True)