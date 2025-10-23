import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Tuple, Callable
import copy
import uuid
import datetime
import os
import re
import ast
from dataclasses import dataclass, field
from .phyenv import PhyEnv
from .utils.metrics import evaluate_hypothesis

@dataclass
class ExperimentRunState:
    """Mutable state for the experiment run."""
    current_iteration: int = 0
    best_hypothesis_info: dict = field(default_factory=lambda: {
        "hypothesis": "",
        "score": 0.0,
        "function": "",
        "is_correct": False
    })
    log_file_path: str = ""

    def update_best_hypothesis(self, hypothesis: str, score: float, function: str, is_correct: bool):
        """Updates the best hypothesis if the new one is better or correct."""
        if score > self.best_hypothesis_info["score"] or (is_correct and not self.best_hypothesis_info["is_correct"]):
            self.best_hypothesis_info["hypothesis"] = hypothesis
            self.best_hypothesis_info["score"] = score
            self.best_hypothesis_info["function"] = function
            self.best_hypothesis_info["is_correct"] = is_correct
            print(f"  ✓ New best hypothesis! Score: {score:.4f}" + (" (Correct!)" if is_correct else ""))
            return True
        print(f"  ✗ Not better than current best hypothesis (score: {self.best_hypothesis_info['score']:.4f})")
        return False


def _log_event(log_file_path: str, event_type: str, data: dict):
    """Helper function to log experiment events to the specified log file."""
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": event_type,
        "data": data
    }
    try:
        with open(log_file_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except IOError as e:
        print(f"Error writing to log file {log_file_path}: {e}")


def setup_logging(env_id: int, model_name: str, history_folder: str, experiment_config: dict = None) -> tuple[str, callable]:
    """
    Sets up logging and returns the log file path and a logging function.
    
    Args:
        env_id: The environment ID
        model_name: The name of the model being used
        history_folder: The folder to store history files
        experiment_config: Optional configuration details to log
    
    Returns:
        A tuple containing (log_file_path, logger_function)
    """
    os.makedirs(history_folder, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"{history_folder}/experiment_{env_id}_{timestamp}_log.jsonl"
    print(f"Full experiment log will be saved to: {log_file_path}")

    # Curry the log_event function with the log_file_path
    def logger(event_type: str, data: dict):
        _log_event(log_file_path, event_type, data)

    setup_data = {
        "env_id": env_id,
        "model": model_name,
        "timestamp": timestamp
    }
    
    # Add experiment configuration if provided
    if experiment_config:
        setup_data.update(experiment_config)
        
    logger("experiment_setup", setup_data)
    return log_file_path, logger


class ResearchInterface:
    """
    ResearchInterface class for running physics experiments with the PhyEnv environment.
    
    This class allows running experiments with input samples, tracking experiment history,
    and managing a sample quota. It can handle both input parameters and dummy variables.
    """
    
    def __init__(self, env: Union[PhyEnv, int, str], sample_quota: int = 200, test_quota: int = 2, mode: str = "default",
                 history_file: Optional[str] = None):
        """
        Initialize an Experiment instance.
        
        Args:
            env: Either a PhyEnv instance or an ID (int or str) to create a PhyEnv instance.
            sample_quota: The maximum number of samples allowed for this experiment.
            history_file: Optional path to a file where experiment history will be saved.
                          If provided, history will be loaded from this file if it exists,
                          and new observations will be appended to it.
        
        Raises:
            ValueError: If the environment is invalid.
        """
        # Set up the environment
        if isinstance(env, PhyEnv):
            self.env = env
        else:
            # Create a PhyEnv instance from the ID
            try:
                self.env = PhyEnv(env)
            except Exception as e:
                raise ValueError(f"Failed to create PhyEnv from ID {env}: {e}")
        
        # Initialize experiment settings
        self.sample_quota = sample_quota
        self.samples_used = 0
        self.test_quota = test_quota
        self.history_file = history_file
        
        # Get all possible parameter names (both required and dummy) and handle anonymization
        self.input_params = self.env.parameter_names
        self.dummy_params = list(self.env.dummy_variables_des.keys()) if hasattr(self.env, 'dummy_variables_des') else []
        self.all_params = self.input_params + self.dummy_params
        self.param_mapping = {param: param for param in self.all_params}
        self._set_mode(mode)
        
        # Initialize observation history
        self.observations = []
        
        # Initialize hypothesis history
        self.tested_hypothesis = []
        
        # Load existing history if available
        if history_file:
            # First try to load from JSONL file
            base_filename = history_file.rsplit('.', 1)[0] if '.' in history_file else history_file
            jsonl_file = f"{base_filename}.jsonl"
            
            try:
                # Try JSONL format first
                if os.path.exists(jsonl_file):
                    with open(jsonl_file, 'r') as f:
                        for line in f:
                            try:
                                record = json.loads(line)
                                if record.get("type") == "observation" and "data" in record:
                                    self.observations.append(record["data"])
                                elif record.get("type") == "hypothesis" and "data" in record:
                                    self.tested_hypothesis.append(record["data"])
                                    self.test_quota -= 1
                            except json.JSONDecodeError:
                                continue
                    
                    self.samples_used = len(self.observations)
                    print(f"Loaded {self.samples_used} existing observations from {jsonl_file}")
            
            except:
                # No history file or invalid JSON, start with empty history
                raise ValueError(f"Failed to load history from {jsonl_file}. Starting with empty history.")
        
        # Set experiment metadata
        self.experiment_id = str(uuid.uuid4())
        self.start_time = datetime.datetime.now().isoformat()
        self.metadata = {
            "problem_id": self.env.id,
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "sample_quota": sample_quota,
            "input_params": self.input_params,
            "dummy_params": self.dummy_params
        }

    def __str__(self) -> str:
        """
        Get a string representation of the Experiment.
        
        Returns:
            A string representation of the Experiment.
        """
        return (f"Experiment(problem_id={self.env.id}, "
                f"samples_used={self.samples_used}/{self.sample_quota})")

    def _set_mode(self, mode: str):
        """
        Set the mode of the experiment.
        Allow the combination of anonymous and no_context/no_description modes.
        Args:
            mode: The mode to set for the experiment.
        """
        self.problem_content = self.env.problem_content
        self.controllable_variables = self.env.controllable_variables_des
        self.observable_var = self.env.output_variable_des
        self.equation = self.env.equation
        if "no_context" in mode:
            self.problem_content = "Unknown context."
        if "no_description" in mode:
            self.problem_content = "Unknown context."
            self.controllable_variables = {key:"Some variable." for key in self.controllable_variables}
            self.observable_var = {key:"Some variable." for key in self.observable_var}
        if "anonymous" in mode:
            self._anonymize_env()
        return

    def _anonymize_env(self):
        """
        Anonymize the environment by replacing variable names with generic ones.
        Replaces input parameters with 'var_1', 'var_2', 'var_3', etc., and updates all
        relevant references in the Interface, including the environment function and equation.
        """
        # Create mapping from original param names to anonymous names
        # Using var_1, var_2, etc. avoids conflicts with existing variables
        param_mapping = {}
        param_reverse = {}
        for i, param in enumerate(self.all_params):
            param_mapping[param] = f"var_{i+1}"
            param_reverse[f"var_{i+1}"] = param
        
        self.param_mapping = param_reverse
        # Update params
        original_input_params = self.input_params.copy()
        original_dummy_params = self.dummy_params.copy()
        self.input_params = [param_mapping[param] for param in original_input_params]
        self.dummy_params = [param_mapping[param] for param in original_dummy_params]
        self.all_params = self.input_params + self.dummy_params
        
        # Update controllable_variables with new names
        anonymized_controllable = {}
        for param, desc in self.controllable_variables.items():
            if param in param_mapping:
                anonymized_controllable[param_mapping[param]] = desc
        self.controllable_variables = anonymized_controllable
        
        # Update observable_var with new names (retain keys but anonymize descriptions)
        self.observable_var = {"var_obs": list(self.observable_var.values())[0]}

        # Update the environment equation using the same mapping
        anonymized_equation = self.equation
        for old_param, new_param in param_mapping.items():
            # Replace parameter in equation (whole word matches only)
            anonymized_equation = re.sub(rf"\b{re.escape(old_param)}\b", new_param, anonymized_equation)
        
        self.equation = anonymized_equation
        
        print(f"Environment anonymized. Variables renamed to: {', '.join(self.all_params)}, {self.observable_var}")

    # --- Running experiments ---
    def run_experiment(self, input_samples: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Run an experiment with a list of input samples.
        
        Args:
            input_samples: A list of dictionaries, where each dictionary contains values
                           for parameters. The dictionaries may include both required input
                           parameters and dummy parameters.
        Returns:
            A dictionary containing:
                - results: List of dictionaries containing the results of each experiment run.
        
        Raises:
            ValueError: If the sample quota has been exceeded
        """
        # Check if we have enough quota left
        if self.samples_used + len(input_samples) > self.sample_quota:
            raise ValueError(
                f"Sample quota exceeded. Used: {self.samples_used}, "
                f"Requested: {len(input_samples)}, Quota: {self.sample_quota}"
            )
        results = []
        
        # Process the experiment samples
        for sample_idx, sample in enumerate(input_samples):
            # Extract only the input parameters needed for execution
            # Map from interface parameter names to environment parameter names
            input_params = {self.param_mapping[k]: v for k, v in sample.items() if k in self.input_params}
            
            # Check if all required parameters are present
            missing_params = set(self.env.parameter_names) - set(input_params.keys())
            if missing_params:
                result = {
                    "sample_id": self.samples_used + sample_idx + 1,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "input": sample,
                    "error": f"Missing required parameters: {missing_params}",
                    "output": None
                }
            else:
                try:
                    # Execute the function with the input parameters
                    output = self.env.execute(**input_params)
                    
                    # Create result dictionary
                    result = {
                        "sample_id": self.samples_used + sample_idx + 1,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "input": sample,
                        "output": output
                    }
                except Exception as e:
                    # Handle execution errors
                    result = {
                        "sample_id": self.samples_used + sample_idx + 1,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "input": sample,
                        "error": str(e),
                        "output": None
                    }
            
            # Add to results
            results.append(result)
            
            # Add to observation history
            self.observations.append(result)
        
        # Update samples used count
        self.samples_used += len(input_samples)
            
        # Save history
        if self.history_file:
            self._save_history()
        
        return results

    # --- Hypothesis testing ---
    def test_hypothesis(self, candidate_function: str, candidate_expr: str) -> Dict[str, Any]:
        """
        Test a candidate function against the observed data and the true environment function.
        
        Args:
            candidate_function: String representation of a Python function to test.
        
        Returns:
            Dictionary containing evaluation metrics.
        
        Raises:
            ValueError: If the candidate function is invalid or cannot be created.
        """           
        # Evaluate the hypothesis
        evaluation = evaluate_hypothesis(
            candidate_function,
            self.env.python_code,
            hyp_expr=candidate_expr,
            true_expr=self.equation,
            observations=self.observations,
            param_names=self.input_params
        )
        
        # Record the hypothesis and its evaluation
        hypothesis_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "hypothesis_id": len(self.tested_hypothesis) + 1,
            "function_code": candidate_function,
            "evaluation": evaluation,
            "num_observations": len(self.observations)
        }
        
        # Add to hypothesis history
        self.tested_hypothesis.append(hypothesis_record)
        
        return evaluation

    def format_hypothesis_function(self, hypothesis: str) -> str:
        """
        Formats a hypothesis expression into a proper Python function.
        
        Args:
            hypothesis: The mathematical expression for the hypothesis
            
        Returns:
            A string representing a Python function implementing the hypothesis
        """
        input_params = [param for param in self.all_params if param in hypothesis]
        function_str = f"""
def hypothesis_function({', '.join(input_params)}):
    return {hypothesis}
"""
        return function_str
    
    def _is_valid_hypothesis_expr(self, expr: str) -> bool:
        if not expr or not isinstance(expr, str):
            return False
        if expr.strip().lower() in ('none', ''):
            return False
        hypo_function_str = self.format_hypothesis_function(expr)
        try:
            ast.parse(hypo_function_str)
        except (SyntaxError, ValueError, TypeError):
            return False
        return True

    def evaluate_hypothesis_with_logging(
        self, 
        hypothesis_expr: str, 
        run_state: ExperimentRunState, 
        logger: Optional[Callable[[str, dict], None]] = None
    ) -> Optional[Dict[str, Any]]:
            
        hypothesis_function_str = self.format_hypothesis_function(hypothesis_expr)
        eval_metrics = None
        
        eval_metrics = self.test_hypothesis(
            candidate_function=hypothesis_function_str,
            candidate_expr=hypothesis_expr
        )
            
        if not eval_metrics: # Should not happen if test_hypothesis raises error, but as a safeguard
            print(f"Hypothesis testing for '{hypothesis_expr}' returned no metrics.")
            if logger:
                logger("hypothesis_testing_failed", {
                    "iteration": run_state.current_iteration,
                    "hypothesis": hypothesis_expr,
                    "reason": "No evaluation metrics returned"
                })
            return None

        score = eval_metrics.get('overall_score', 0.0)
        is_correct = eval_metrics.get('is_correct', False)
        
        updated = run_state.update_best_hypothesis(hypothesis_expr, score, hypothesis_function_str, is_correct)

        if logger:
            logger("hypothesis_testing", {
                "iteration": run_state.current_iteration,
                "hypothesis": hypothesis_expr,
                "hypothesis_function": hypothesis_function_str, # Can be large
                "evaluation": eval_metrics,
                "is_best_so_far": updated,
                "is_correct": is_correct
            })
            
            if is_correct:
                logger("correct_hypothesis_found", {
                    "iteration": run_state.current_iteration,
                    "hypothesis": hypothesis_expr,
                    "score": score,
                    "function": hypothesis_function_str # Can be large
                })
        
        return eval_metrics

    # --- Getting history and current state ---
    def get_observations(self, as_pandas: bool = False) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """
        Get the observation history.
        
        Args:
            as_pandas: If True, returns a pandas DataFrame instead of a list of dictionaries.
        
        Returns:
            The observation history as either a list of dictionaries or a pandas DataFrame.
        """
        if as_pandas:
            # Convert to pandas DataFrame
            df = pd.DataFrame(self.observations)
            
            # Expand the input column if it exists
            if 'input' in df.columns:
                input_df = pd.json_normalize(df['input'])
                # Drop the original input column
                df = df.drop('input', axis=1)
                # Join the expanded input columns
                df = pd.concat([df, input_df], axis=1)
            
            return df
        else:
            return copy.deepcopy(self.observations)
    
    def get_hypothesis_history(self, as_pandas: bool = False) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """
        Get the hypothesis testing history.
        
        Args:
            as_pandas: If True, returns a pandas DataFrame instead of a list of dictionaries.
        
        Returns:
            The hypothesis history as either a list of dictionaries or a pandas DataFrame.
        """
        if as_pandas:
            # Convert to pandas DataFrame
            if self.tested_hypothesis:
                df = pd.DataFrame(self.tested_hypothesis)
                
                # Expand evaluation metrics if possible
                if 'evaluation' in df.columns:
                    # Extract key metrics for easier analysis
                    df['fit_quality'] = df['evaluation'].apply(lambda x: x.get('fit_quality', 0))
                    df['equivalence_score'] = df['evaluation'].apply(lambda x: x.get('equivalence_score', 0))
                    df['overall_score'] = df['evaluation'].apply(lambda x: x.get('overall_score', 0))
                    df['is_correct'] = df['evaluation'].apply(lambda x: x.get('is_correct', False))
                    df['fits_data'] = df['evaluation'].apply(lambda x: x.get('fits_data', False))
                
                return df
            else:
                return pd.DataFrame()
        else:
            return copy.deepcopy(self.tested_hypothesis)
    
    def get_remaining_quota(self) -> int:
        """
        Get the remaining sample quota.
        
        Returns:
            The number of samples remaining in the quota.
        """
        return self.sample_quota - self.samples_used

    # --- Saving history ---
    def _save_history(self):
        """
        Save the observation history to a JSONL file to properly handle boolean variables.
        Each line in the file contains a single JSON record.
        """
        if not self.history_file:
            return
        try:
            # Get base filename without extension
            base_filename = self.history_file.rsplit('.', 1)[0] if '.' in self.history_file else self.history_file
            jsonl_file = f"{base_filename}.jsonl"
            
            with open(jsonl_file, 'w') as f:
                # Write metadata as first line
                f.write(json.dumps({"type": "metadata", "data": self.metadata}) + '\n')
                
                # Write each observation as a separate line
                for obs in self.observations:
                    f.write(json.dumps({"type": "observation", "data": obs}) + '\n')
                
                # Write each hypothesis record as a separate line
                for hyp in self.tested_hypothesis:
                    f.write(json.dumps({"type": "hypothesis", "data": hyp}) + '\n')
                
            print(f"History saved to {jsonl_file}")
                
        except:
            raise ValueError(f"Failed to save history to {jsonl_file}.")
    
    def generate_report(self) -> str:
        """
        Generate a report of the experiment.
        
        Returns:
            A string containing a report of the experiment.
        """
        report = []
        report.append(f"Experiment Report (ID: {self.experiment_id})")
        report.append(f"Problem ID: {self.env.id}")
        report.append(f"Start Time: {self.start_time}")
        report.append(f"Samples Used: {self.samples_used} / {self.sample_quota}")
        
        # Add parameter information
        report.append("\nInput Parameters:")
        for param in self.input_params:
            desc = self.env.get_param_description(param)
            report.append(f"  {param}: {desc}")
        
        if self.dummy_params:
            report.append("\nDummy Parameters (defined but not used):")
            for param in self.dummy_params:
                desc = self.env.get_param_description(param)
                report.append(f"  {param}: {desc}")
        
        # Add hypothesis testing information
        if self.tested_hypothesis:
            report.append("\nHypothesis Testing:")
            report.append(f"  Number of hypotheses tested: {len(self.tested_hypothesis)}")
            
            # Find the best hypothesis
            best_hypothesis = max(self.tested_hypothesis, 
                                 key=lambda h: h["evaluation"].get("overall_score", 0))
            
            report.append(f"  Best hypothesis (ID: {best_hypothesis['hypothesis_id']}):")
            report.append(f"    Overall score: {best_hypothesis['evaluation'].get('overall_score', 0):.4f}")
            report.append(f"    Fit quality: {best_hypothesis['evaluation'].get('fit_quality', 0):.4f}")
            report.append(f"    Equivalence score: {best_hypothesis['evaluation'].get('equivalence_score', 0):.4f}")
            report.append(f"    Is correct: {best_hypothesis['evaluation'].get('is_correct', False)}")
            report.append(f"    Fits data: {best_hypothesis['evaluation'].get('fits_data', False)}")
            
            # Include the function code for reference
            func_code = best_hypothesis['function_code'].strip()
            if len(func_code) > 500:  # Truncate long functions
                func_lines = func_code.split('\n')
                if len(func_lines) > 10:
                    short_code = '\n'.join(func_lines[:5] + ['...'] + func_lines[-5:])
                else:
                    short_code = func_code[:500] + '...'
                report.append(f"    Function (truncated): \n{short_code}")
            else:
                report.append(f"    Function: \n{func_code}")
        
        return "\n".join(report)
    
    def generate_final_report(
        self, 
        run_state: ExperimentRunState, 
        logger: callable = None
    ) -> dict:
        """
        Generates and logs the final experiment report.
        
        Args:
            run_state: The experiment run state
            logger: Optional logging function
            
        Returns:
            Dictionary with the final results
        """
        print(f"\n{'='*60}")
        print(f"Final Results after {run_state.current_iteration} iterations")
        print(f"{'='*60}")

        best_info = run_state.best_hypothesis_info
        if best_info["hypothesis"]:
            print(f"Best hypothesis found: {best_info['hypothesis']}")
            print(f"Score: {best_info['score']:.4f}")
            print(f"Function:\n{best_info['function']}")
            if best_info["is_correct"]:
                print("This hypothesis was confirmed as correct.")
        else:
            print("No valid hypothesis was confirmed or significantly scored during the run.")

        print("\nGenerating full experiment report...")
        report = self.generate_report()
        print(report)  # Print report to console

        if logger:
            logger("experiment_complete", {
                "iterations_completed": run_state.current_iteration,
                "best_hypothesis": best_info['hypothesis'],
                "best_hypothesis_score": best_info['score'],
                "best_hypothesis_function": best_info['function'],
                "best_hypothesis_is_correct": best_info['is_correct'],
                "final_sample_quota_remaining": self.get_remaining_quota(),
                "final_test_quota_remaining": self.test_quota,
                "experiment_report_summary": report,
                "log_file": run_state.log_file_path,
                "end_time": datetime.datetime.now().isoformat()
            })
            print(f"\nExperiment complete! Full log saved to: {run_state.log_file_path}")
            
        # Return final results for potential external use
        return {
            "best_hypothesis": best_info["hypothesis"],
            "score": best_info["score"],
            "function": best_info["function"],
            "is_correct": best_info["is_correct"],
            "log_file": run_state.log_file_path,
            "iterations_completed": run_state.current_iteration,
            "final_sample_quota_remaining": self.get_remaining_quota(),
            "final_test_quota_remaining": self.test_quota
        }
   
    # --- For baseline_experiment ---
    def execute_experiments_and_evaluate(
        self, 
        experiments_to_run: list, 
        current_hypothesis: str,
        test_flag: bool,
        run_state: ExperimentRunState, 
        logger: callable = None
    ) -> dict:
        """
        Runs the proposed experiments and evaluates the hypothesis if conditions are met.
        
        Args:
            experiments_to_run: List of experiment parameters to run
            current_hypothesis: The hypothesis expression to evaluate if test_flag is True
            test_flag: Whether to test the hypothesis
            run_state: The current experiment run state
            logger: Optional logging function
            
        Returns:
            Experiment results data
        """
        # Part 1: Evaluate hypothesis
        # Check if hypothesis is a valid expression
        valid_hypothesis_provided = self._is_valid_hypothesis_expr(current_hypothesis)
        is_tested = False
        if valid_hypothesis_provided:
            if test_flag and self.test_quota > 0:
                print(f"\nTesting hypothesis: {current_hypothesis}")
                self.test_quota -= 1
                
                # Use the refactored evaluate_hypothesis_with_logging
                self.evaluate_hypothesis_with_logging(current_hypothesis, run_state, logger)
                is_tested = True

                if run_state.best_hypothesis_info["is_correct"]:
                    print("Correct hypothesis found. Skipping further experiments in this cycle if configured.")
                    return None 
                    
            elif test_flag: # test_flag is true, hypothesis exists, but cannot test (e.g. quota)
                print(f"\nHypothesis '{current_hypothesis}' was provided but not tested (Test quota: {self.test_quota}).")
        
            hypothesis_info = {
                "current_hypothesis": current_hypothesis,
                "is_tested": is_tested,
                "time_stamp": datetime.datetime.now().isoformat()
            }
            self.observations.append(hypothesis_info)
            
        # Part 2: Execute experiments
        if experiments_to_run:
            print(f"\nRunning {len(experiments_to_run)} experiments...")
            experiment_results_data = self.run_experiment(experiments_to_run)

            if logger:
                logger("experiment_execution", {
                    "iteration": run_state.current_iteration,
                    "experiments_count": len(experiments_to_run),
                    "experiments": experiments_to_run,
                    "results": experiment_results_data
                })

            print("\nExperiment Results:")
            for res in experiment_results_data:
                print(f"  Sample {res['sample_id']}: Input: {res['input']}, Output: {res['output']}" +
                    (f", Error: {res['error']}" if 'error' in res else ""))
        else:
            print("\nNo experiments to run in this iteration.")
            experiment_results_data = None
        
        return experiment_results_data
    
    # --- For LLM agents ---
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute a JSON-like text action and return environment response for LLM agents.
        
        Args:
            action: JSON-like string action, e.g.:
                "{'type': 'run_experiment', 'params': {'m': 1.5, 'h': 2.0}}"
                "{'type': 'test_hypothesis', 'hypothesis': 'm * 9.8 * h'}"
                "{'type': 'get_status'}"
        
        Returns:
            observation: Text description of results
            reward: Scalar reward based on progress  
            done: Whether episode is complete (correct hypothesis or quotas exhausted)
            info: Additional metadata dictionary
        """       
        # Parse the JSON-like action string
        try:
            action_dict = ast.literal_eval(action)
            action_type = action_dict.get('type', '')
        except (SyntaxError, ValueError) as e:
            observation = f"Error parsing action: {str(e)}. Expected format: {{'action': 'action_type', 'params': {{}}}}"
            reward = -0.1
            done = False
            info = {'error': 'parse_error'}
            return observation, reward, done, info
            
        # Initialize response variables
        observation = ""
        reward = 0.0
        done = False
        info = {
            'action_type': action_type,
            'remaining_sample_quota': self.get_remaining_quota(),
            'remaining_test_quota': self.test_quota
        }
        
        if action_type == 'run_experiment':
            params = action_dict.get('params', {})
            if not params:
                observation = "Error: No parameters provided for experiment"
                reward = -0.1
            else:
                results = self.run_experiment([params])
                if results and len(results) > 0:
                    result = results[0]
                    if 'error' in result:
                        observation = f"Experiment failed: {result['error']}"
                        reward = -0.1
                    else:
                        observation = f"Input: {result['input']}, Output: {result['output']}"
                        reward = 0.1
                        info['last_experiment_result'] = result
                else:
                    observation = "No experiment results returned"
                    reward = -0.1
        
        elif action_type == 'test_hypothesis':
            hypothesis = action_dict.get('hypothesis', '')
            if not hypothesis:
                observation = "Error: No hypothesis provided"
                reward = -0.1
            else:
                # Format the hypothesis as a function
                hypothesis_function = self.format_hypothesis_function(hypothesis)
                evaluation = self.test_hypothesis(hypothesis_function, hypothesis)
                
                # Decrease test quota
                self.test_quota -= 1
                
                score = evaluation.get('overall_score', 0.0)
                is_correct = evaluation.get('is_correct', False)
                fits_data = evaluation.get('fits_data', False)
                
                if is_correct:
                    observation = f"Hypothesis CORRECT! Score: {score:.4f}. You found the true equation!"
                    reward = 10.0  # Large reward for correct hypothesis
                    done = True
                elif fits_data:
                    observation = f"Hypothesis fits the data but may not be the true equation. Score: {score:.4f}"
                    reward = score * 2.0  # Reward proportional to score
                else:
                    observation = f"Hypothesis does not fit the data well. Score: {score:.4f}"
                    reward = score * 0.5  # Small reward for any progress
                
                info['last_hypothesis_evaluation'] = evaluation
                info['hypothesis_score'] = score
                info['hypothesis_correct'] = is_correct
        
        elif action_type == 'get_status':
            observation = f"""Current Status:
- Sample quota: {self.get_remaining_quota()}/{self.sample_quota} remaining
- Test quota: {self.test_quota} remaining  
- Experiments conducted: {len(self.observations)}
- Hypotheses tested: {len(self.tested_hypothesis)}"""
            reward = 0.0  # No reward for status check
        
        else:
            observation = f"Error: Unknown action type '{action_type}'. Valid actions: run_experiment, test_hypothesis, get_status"
            reward = -0.1
        
        # Check termination conditions
        if self.get_remaining_quota() <= 0 and self.test_quota <= 0:
            done = True
            observation += " (All quotas exhausted - episode terminated)"
        
        return observation, reward, done, info


# Example usage:
if __name__ == "__main__":
    # Create an experiment with PhyEnv ID 285
    exp = ResearchInterface(285, sample_quota=10)
    
    # Print experiment information
    exp._anonymize_env()
    print(exp)

    # Generate some random input samples
    input_samples = []
    for _ in range(5):
        sample = {}
        for param in exp.all_params:
            sample[param] = np.random.uniform(0.1, 10.0)
        input_samples.append(sample)
    
    # Run the experiment
    experiment_result = exp.run_experiment(input_samples)
    results = experiment_result
    
    # Print the results
    print("\nExperiment Results:")
    for result in results:
        print(f"  Sample {result['sample_id']}: {result['output']}")
    
    # Example of testing a hypothesis
    # This example assumes the environment function is a simple formula
    candidate_function = """
def hypothesis_function(g, r):
a = g * math.sqrt(r)
return a
"""
    
    # Print the experiment report
    print("\n" + exp.generate_report())