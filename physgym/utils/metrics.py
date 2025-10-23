import numpy as np
import pandas as pd
import inspect
from typing import Callable, Dict, List, Any, Union, Optional
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from scipy.stats import kendalltau, pearsonr
import sympy as sp
from tqdm import tqdm
from physgym.utils.llm_providers import generate_with_provider, load_api_key
from physgym.utils.sandbox import create_function_from_string
import json
import re
from types import SimpleNamespace

# Define constants and tolerance
TOLERANCE = 1e-5
CONSTANTS_TO_CHECK = {
    sp.pi: sp.pi.evalf(20),
    sp.E: sp.E.evalf(20)
}

def _substitute_numerical_approximations(expr):
    """
    Walks through an expression and replaces numerical approximations of constants
    (like 3.14159) with their symbolic sympy counterparts (like pi).
    """
    for number in expr.atoms(sp.Number):
        for symbolic_const, float_val in CONSTANTS_TO_CHECK.items():
            if sp.Abs(number - float_val) < TOLERANCE:
                expr = expr.subs(number, symbolic_const)
    return expr

def try_symbolic_equivalence(func1_str: str, func2_str: str, param_names: List[str], assumptions: Optional[Dict[str, bool]] = {'real': True}) -> Dict[str, Any]:
    """
    Attempt to prove symbolic equivalence between two function expressions,
    including expressions with common numpy functions (e.g., np.sin, np.exp)
    and constants (np.pi).
    
    Args:
        func1_str: String representation of the first function expression.
        func2_str: String representation of the second function expression.
        param_names: List of parameter names (variables) used in the expressions.
        
    Returns:
        A dictionary containing the symbolic equivalence results.
    """
    print("Attempting symbolic equivalence verification...")
    try:
        # Define a mapping from numpy names to sympy functions/constants
        numpy_to_sympy_map = {
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'asin': sp.asin,
            'acos': sp.acos,
            'atan': sp.atan,
            'exp': sp.exp,
            'log': sp.log,
            'sqrt': sp.sqrt,
            'pi': sp.pi,
            'e': sp.E,
            'abs': sp.Abs,
        }
        # Use SimpleNamespace to mimic module access (e.g., np.sin)
        sympy_namespace = SimpleNamespace(**numpy_to_sympy_map)

        # Create symbols with optional assumptions
        symbol_assumptions = assumptions or {}
        symbols = {name: sp.Symbol(name, real=symbol_assumptions.get(name, True)) for name in param_names}

        # Create the local namespace for parsing the strings
        local_namespace = {
            'np': sympy_namespace,
            'numpy': sympy_namespace, # Allow 'numpy.' as well as 'np.'
            **symbols
        }
        
        # Parse expressions using the local namespace
        expr1 = sp.sympify(func1_str, locals=local_namespace)
        expr2 = sp.sympify(func2_str, locals=local_namespace)

        # Substitute numerical approximations
        expr1 = _substitute_numerical_approximations(expr1)
        expr2 = _substitute_numerical_approximations(expr2)
        
        # Check if the simplified difference is zero
        difference = sp.simplify(expr1 - expr2)
        is_equivalent = (difference == 0)
        
        if is_equivalent:
            print("✓ Functions are symbolically equivalent")
        else:
            print("✗ Functions are not symbolically equivalent")
            print(f"  Difference: {difference}")
            
        return {
            "is_equivalent": bool(is_equivalent),
            "symbolic_diff": str(difference),
            "method": "symbolic"
        }
    except Exception as e:
        print(f"✗ Symbolic verification failed: {e}")
        return {
            "is_equivalent": False,
            "error": str(e),
            "method": "symbolic"
        }


def evaluate_hypothesis_fit(hypothesis_func: Callable, observations: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate how well a hypothesis function fits the observed data points.
    
    Args:
        hypothesis_func: The candidate function to evaluate
        observations: List of observation dictionaries with 'input' and 'output' keys
        
    Returns:
        Dictionary of metrics evaluating the fit
    """
    # Get function signature to determine required parameters
    sig = inspect.signature(hypothesis_func)
    param_names = list(sig.parameters.keys())
    
    # Filter out observations with errors
    valid_observations = []
    for obs in observations:
        if 'NaN' not in obs and isinstance(obs.get('output'), (int, float)):
            valid_observations.append(obs)
    
    if not valid_observations:
        print("No valid observations found for evaluation.")
        return {
            "mse": float('inf'),
            "nmse": float('inf'),
            "r2": float('-inf'),
            "pearson_r": 0.0,
            "kendall_tau": 0.0,
            "mape": float('inf'),
            "fit_quality": 0.0,
            "valid_points": 0
        }
    
    # Prepare data for evaluation
    y_true = []
    y_pred = []
    
    # Use tqdm to show progress
    print("Evaluating fit to observed data:")
    for obs in tqdm(valid_observations, desc="Evaluating fit", unit="obs"):
        if 'error' in obs or isinstance(obs["output"], str) or obs["output"] is None or np.isnan(obs["output"]):
            # Skip observations with errors
            continue
        input_params = obs['input']
        # Extract parameters needed by the hypothesis function
        args = {param: input_params[param] for param in param_names if param in input_params}
        # Check if we have all required parameters
        if len(args) == len(param_names):
            # Predict output using hypothesis function
            try:
                predicted = hypothesis_func(**args)
            except Exception as e:
                print(f"Error during hypothesis function execution: {e}")
                continue
            if np.isnan(predicted) or np.isinf(predicted):
                continue
            # Add to comparison arrays
            y_true.append(obs['output'])
            y_pred.append(predicted)
    
    # Convert to numpy arrays for calculations
    y_true_clean = np.array(y_true)
    y_pred_clean = np.array(y_pred)

    # If no valid predictions, return default metrics
    if len(y_true) == 0:
        print("No valid predictions found for evaluation.")
        return {
            "mse": float('inf'),
            "nmse": float('inf'),
            "r2": float('-inf'),
            "pearson_r": 0.0,
            "kendall_tau": 0.0,
            "mape": float('inf'),
            "fit_quality": 0.0,
            "valid_points": 0
        }
    
    # Compute basic metrics
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    
    # Compute variance of actual values for normalized MSE
    var = np.var(y_true_clean)
    
    # Avoid division by zero for normalized MSE
    nmse = mse / var if var > 0 else float('inf')
    
    # Compute R² (coefficient of determination)
    if np.sum((y_true_clean - y_true_clean.mean())**2) > 0:
        r2 = r2_score(y_true_clean, y_pred_clean)
        r2 = max(-10, r2)  # Clamp R² to prevent extreme negative values
    else:
        r2 = 0  # R² is undefined when all actual values are identical

    # Compute Kendall's tau (rank correlation)
    kendall = kendalltau(y_true_clean, y_pred_clean)[0]
    kendall = 0.0 if np.isnan(kendall) else kendall
    
    # Compute Mean Absolute Percentage Error
    # Avoid division by zero in MAPE calculation
    valid_mape_indices = y_true_clean != 0
    if np.any(valid_mape_indices):
        mape = mean_absolute_percentage_error(
            y_true_clean[valid_mape_indices], 
            y_pred_clean[valid_mape_indices]
        )
    else:
        mape = float('inf')
    
    # Compute overall fit quality (combination of metrics)
    # Scale from 0-1 where 1 is perfect fit
    r2_component = (r2 + 1) / 2 if r2 > -1 else 0  # Transform R² from [-1,1] to [0,1]
    kendall_component = abs(kendall)  # Use absolute rank correlation
    
    # Geometric mean of components for overall quality
    fit_quality = (r2_component * kendall_component) ** (1/2)
    
    return {
        "mse": mse,
        "nmse": nmse,
        "r2": r2,
        "kendall_tau": kendall,
        "mape": mape,
        "fit_quality": fit_quality,
        "valid_points": len(y_true_clean)
    }


def check_function_equivalence_llm(func1_str: str, func2_str: str, 
                                   model: str = "google/gemini-2.5-flash",
                                   provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Check if two function strings are mathematically equivalent using LLM as a judge.
    The function extracts the body of each function and compares them.
    
    Args:
        func1_str: String representation of the first function (e.g., the true environment function)
        func2_str: String representation of the second function (e.g., the hypothesis function)
        model: LLM model to use for equivalence checking
        provider: LLM provider (auto-detected if None)
        
    Returns:
        Dictionary with equivalence metrics including LLM judgment
    """
    print("Checking function equivalence using LLM...")
    
    # Extract function bodies, stripping out function declaration and other non-essential parts
    def extract_function_body(func_str, is_return=True):
        if is_return:
            # If we are looking for a return statement, we can try to find it directly
            return_match = re.search(r'return\s+(.+?)(?:[;\n#]|$)', func_str, re.DOTALL)
            if return_match:
                return return_match.group(1).strip()
        # Try to find the body of the function (between outermost opening and closing braces)
        body_match = re.search(r'def\s+\w+\s*\([^)]*\)\s*(?:\s*->\s*[^:]+)?\s*:(.*?)(?:$|(?=\ndef\s+\w+))', func_str, re.DOTALL)
        if body_match:
            body = body_match.group(1).strip()
            return body  # Return full body
        return func_str  # Return original if parsing fails
    
    # Extract function bodies and parameter names
    # func1_body = extract_function_body(func1_str)
    # func2_body = extract_function_body(func2_str)

    # We assume the function strings are already expressions, not full function definitions
    func1_body = func1_str
    func2_body = func2_str

    # Load the prompt template from package
    import os
    from pathlib import Path
    package_dir = Path(__file__).parent.parent
    prompt_path = package_dir / "prompts" / "equivalence_check.txt"
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    # Use safe string replacement instead of format
    prompt = prompt_template.replace("{function1}", func1_body).replace("{function2}", func2_body)
    
    # Call the LLM to evaluate equivalence
    try:
        system_message = "You are an expert physicist and mathematician who can determine whether two mathematical expressions are equivalent."
        llm_response, _ = generate_with_provider(
            prompt=prompt,
            model=model,
            provider=provider,
            system_message=system_message,
            temperature=0.2  # Low temperature for more deterministic reasoning
        )
        
        # Extract JSON from response
        try:
            # Try to find JSON object in the response
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else:
                # If no JSON found, try to parse the whole response
                result = json.loads(llm_response)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON from LLM response. Using heuristic parsing.")
            
            # Use heuristics to extract information
            are_equivalent = "equivalent" in llm_response.lower() and not "not equivalent" in llm_response.lower()
            confidence = 0.7  # Default confidence when we can't parse accurately
            explanation = llm_response
            differences = "Could not parse differences" if not are_equivalent else ""
            
            result = {
                "are_equivalent": are_equivalent,
                "confidence": confidence,
                "explanation": explanation,
                "differences": differences
            }
        
        # Extract metrics from result
        is_equivalent = result.get("are_equivalent", False)
        confidence = result.get("confidence", 0.0)
        explanation = result.get("explanation", "")
        differences = result.get("differences", "")

        # If confidence is low, we can consider it not equivalent
        if confidence < 0.9:
            is_equivalent = False
            differences = "Low confidence in equivalence, likely not equivalent."
        
        # Map confidence to equivalence score
        equivalence_score = confidence if is_equivalent else (1 - confidence) * 0.3
        
        # Print results for user
        if is_equivalent:
            print(f"✓ Functions are equivalent (confidence: {confidence:.2f})")
        else:
            print(f"✗ Functions are not equivalent (confidence: {confidence:.2f})")
            if differences:
                print(f"  Differences: {differences}")
        
        return {
            "is_equivalent": bool(is_equivalent),
            "equivalence_score": equivalence_score,
            "confidence": confidence,
            "method": "llm"
        }
    
    except Exception as e:
        print(f"Error during LLM equivalence check: {e}")
        return {
            "is_equivalent": False,
            "equivalence_score": 0.0,
            "error": str(e),
            "method": "llm_failed"
        }
    

def evaluate_hypothesis(hypothesis_function: str, true_func_str: str, hyp_expr: str, true_expr: str,
                       observations: List[Dict[str, Any]],
                       param_names: List):
    """
    Comprehensively evaluate a hypothesis function against both 
    observed data and the true underlying function.
    
    Args:
        hypothesis_func: The candidate function to evaluate
        true_func: The true function (environment function)
        observations: List of observation dictionaries
        param_ranges: Dictionary mapping parameter names to (min, max) tuples
        num_test_points: Number of random test points to evaluate for equivalence check
        
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    print("Starting hypothesis evaluation...")
    # Evaluate fit to observed data
    # Create a callable function from the string in a sandbox
    hypothesis_func = create_function_from_string(hypothesis_function, sandbox=True, timeout=5, fast_local=True)
    fit_metrics = evaluate_hypothesis_fit(hypothesis_func, observations)
    
    print("\nFit metrics:")
    print(f"  MSE: {fit_metrics.get('mse', float('inf')):.6f}")
    print(f"  NMSE: {fit_metrics.get('nmse', float('inf')):.6f}")
    print(f"  R²: {fit_metrics.get('r2', float('-inf')):.6f}")
    print(f"  Kendall's tau: {fit_metrics.get('kendall_tau', 0):.6f}")
    print(f"  MAPE: {fit_metrics.get('mape', float('inf')):.6f}")
    print(f"  Fit quality: {fit_metrics.get('fit_quality', 0):.6f}")
    print(f"  Valid points: {fit_metrics.get('valid_points', 0)}")
    
    # Try symbolic equivalence
    symbolically_equivalent = False        
    # Try symbolic equivalence
    sym_result = try_symbolic_equivalence(hyp_expr, true_expr, param_names)
    symbolically_equivalent = sym_result.get("is_equivalent", False)

    # Special numbers case
    need_llm = False
    if getattr(sym_result, 'error', None):
        print(f"Symbolic equivalence check failed: {sym_result['error']}")
        need_llm = True
    if not symbolically_equivalent:
        special_numbers = ["np.pi", "np.e"]
        for number in special_numbers:
            if number in true_expr and number not in hyp_expr:
                need_llm = True
        if fit_metrics.get("mse", float('inf')) < 0.01:
            need_llm = True
    # If symbolic equivalence check fails, fall back to LLM
    if need_llm:
        # Check equivalence using LLM
        equiv_metrics = check_function_equivalence_llm(
            true_expr, hyp_expr, model="google/gemini-2.5-flash", provider="openrouter"
        )
        symbolically_equivalent = equiv_metrics.get("is_equivalent", False)
        symbolically_equivalent = bool(symbolically_equivalent)
        print("\nLLM Evaluation:")
        print(f"  Equivalence score: {equiv_metrics.get('equivalence_score', 0):.6f}")
        print(f"  Is equivalent: {symbolically_equivalent}")
        print(f"  Method: {equiv_metrics.get('method', 'llm')}")
    else:
        equiv_metrics = {
            "is_equivalent": symbolically_equivalent,
            "equivalence_score": 1.0 if symbolically_equivalent else 0.0,
            "confidence": 1.0,
            "method": "symbolic"
        }
    # Combine metrics
    fit_quality = fit_metrics.get("fit_quality", 0)
    equivalence_score = equiv_metrics.get("equivalence_score", 0)

    # Calculate overall score - mean of fit quality and equivalence score
    overall_score = (fit_quality + equivalence_score) * 0.5
    
    combined_metrics = {
        "fit_metrics": fit_metrics,
        "equivalence_metrics": equiv_metrics,
        
        # Summary metrics
        "fit_quality": fit_quality,
        "equivalence_score": equivalence_score,
        
        # Overall score (geometric mean of fit quality and equivalence)
        "overall_score": overall_score,
        
        # Boolean flags
        "is_correct": symbolically_equivalent,
        "fits_data": bool(fit_quality > 0.7)
    }
    
    return combined_metrics