import json
import re
import inspect
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

# Difficulties Criteria: 1. the length of the solution. 2. the complexity of the answer equation. 3. the number of variables.

class PhyEnv:
    """
    Physics Environment class that transforms processed physics problems into executable environments.
    
    This class loads a sample from full_samples.json, extracts the Python code for the
    env_function, makes it executable, and provides access to variable descriptions and metadata.
    """
    
    def __init__(self, sample_data: Union[Dict[str, Any], int, str], samples_file: str = None, static: bool = False):
        """
        Initialize a PhyEnv instance from a sample.
        
        Args:
            sample_data: Either a dictionary containing the sample data, or an ID (int or str) 
                         of a sample to load from the samples_file.
            samples_file: Path to the JSON file containing processed samples.
                          Only used if sample_data is an ID.
        
        Raises:
            ValueError: If the sample data is invalid or the env_function cannot be created.
            FileNotFoundError: If the samples file cannot be found.
            KeyError: If the sample ID is not found in the samples file.
        """
        # Load the sample data
        if isinstance(sample_data, (int, str)):
            # Load from file by ID
            try:
                # Use default path within package if not specified
                if samples_file is None:
                    package_dir = Path(__file__).parent
                    samples_file = package_dir / "samples" / "full_samples.json"
                
                with open(samples_file, 'r') as f:
                    all_samples = json.load(f)
                
                # Find the sample with the matching ID
                sample_id = str(sample_data) if isinstance(sample_data, int) else sample_data
                sample = next((s for s in all_samples if str(s.get("id")) == sample_id), None)
                
                if sample is None:
                    raise KeyError(f"Sample with ID {sample_id} not found in {samples_file}")
                
                self.sample_data = sample
            except FileNotFoundError:
                raise FileNotFoundError(f"Samples file {samples_file} not found")
        elif isinstance(sample_data, dict):
            # Use the provided dictionary
            self.sample_data = sample_data
        else:
            raise ValueError("sample_data must be a dictionary, integer ID, or string ID")
        
        # Extract the Python code
        self.python_code = self.sample_data.get("python_code", "")
        if not self.python_code:
            raise ValueError("No Python code found in the sample data")
        
        # Extract equation and variable descriptions
        self.equation = self.sample_data.get("equation", "")
        self.input_variables_des = self.sample_data.get("input_variables", {})
        self.output_variable_des = self.sample_data.get("output_variable", {})
        self.dummy_variables_des = self.sample_data.get("dummy_variables", {})
        self.controllable_variables_des = self.input_variables_des.copy()
        self.controllable_variables_des.update(self.dummy_variables_des)

        # Metadata
        self.id = self.sample_data.get("id", "unknown")
        self.problem_content = self.sample_data.get("content", "")
        self.answer = self.sample_data.get("answer", "")
        self.tag = self.sample_data.get("tag", "")
        self.level = self.sample_data.get("level", "")
        self.solution = self.sample_data.get("solution", "")
        
        # Create the executable function
        if not static:
            self._create_env_function()
    
    def _create_env_function(self):
        """
        Create an executable env_function from the Python code.
        
        This method extracts the function body, creates a new function with the correct
        signature, and makes it accessible as self.env_function.
        
        Raises:
            ValueError: If the env_function cannot be created.
        """
        # Function creation namespace
        namespace = {
            'np': np,
            'math': __import__('math')
        }
        
        try:
            # Clean up the code and rename the function if needed
            code = self.python_code
            
            # Update function name to env_function if it's not already
            code = re.sub(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', 'def env_function(', code)
            
            # Execute the code in the namespace
            exec(code, namespace)
            
            # Get the function from the namespace
            self.env_function = namespace['env_function']
            
            # Store the function signature and docstring
            self.function_signature = inspect.signature(self.env_function)
            self.function_docstring = self.env_function.__doc__ or ""
            
            # Extract parameter names
            self.parameter_names = list(self.function_signature.parameters.keys())
        
        except Exception as e:
            raise ValueError(f"Failed to create env_function: {e}")
    
    def execute(self, **kwargs) -> float:
        """
        Execute the env_function with the provided parameters.
        
        Args:
            **kwargs: The parameters to pass to the env_function.
        
        Returns:
            The result of the env_function.
        
        Raises:
            ValueError: If the required parameters are not provided or are invalid.
        """
        # Check that all required parameters are provided
        for param in self.parameter_names:
            if param not in kwargs:
                raise ValueError(f"Required parameter '{param}' not provided")
        
        # Execute the function with the provided parameters
        result = self.env_function(**kwargs)
        if isinstance(result, complex):
            # If the result is complex, return it as a string
            result = "Invalid result: complex number"
        return result
    
    def get_param_description(self, param_name: str) -> Optional[str]:
        """
        Get the description of a parameter.
        
        Args:
            param_name: The name of the parameter.
        
        Returns:
            The description of the parameter, or None if not found.
        """
        if param_name in self.input_variables_des:
            return self.input_variables_des[param_name]
        elif param_name in self.dummy_variables_des:
            return self.dummy_variables_des[param_name]
        return None
    
    def __str__(self) -> str:
        """
        Get a string representation of the PhyEnv.
        
        Returns:
            A string representation of the PhyEnv.
        """
        output = f"PhyEnv(id={self.id}, level={self.level}, tag={self.tag})\n"
        output += f"Parameters: {', '.join(self.parameter_names)}\n"
        if self.output_variable_des:
            output_name = next(iter(self.output_variable_des))
            output += f"Output: {output_name}\n"
        output += f"Equation: {self.equation}\n"
        return output


# Example usage:
if __name__ == "__main__":
    # Load a sample from the processed samples
    try:
        samples_file = 'samples/processed_samples.json'
        env = PhyEnv(134, samples_file)  # Load sample with ID 674
        
        # Print information about the environment
        print(env)
        print("\nProblem:")
        print(env.get_problem_content())
        print("\nAnswer formula:")
        print(env.get_answer())
        
        # Print parameter descriptions
        print("\nParameters:")
        for param in env.get_input_params():
            desc = env.get_param_description(param)
            print(f"  {param}: {desc}")
        
        # Print equation
        print("\nEquation:")
        print(env.get_equation())
        
        # Print output description
        output_info = env.output_variable
        if output_info:
            print(f"\nOutput {output_info}")
        
        # Print dummy variables if any
        if env.dummy_variables:
            print("\nDummy Variables (defined but not used):")
            for var, desc in env.dummy_variables.items():
                print(f"  {var}: {desc}")
        
        # Execute the function
        result = env.execute(g=9.8, r=0.01)
        print(f"\nResult: {result}")
        
    except Exception as e:
        print(f"Error: {e}")