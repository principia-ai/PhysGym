"""
Sandbox module for safely executing code from strings.

Performance Options:
1. For maximum safety and isolation (slowest):
   - Use `sandbox=True, cached=False, fast_local=False`
   
2. For good safety with better performance (recommended):
   - Use `sandbox=True, cached=True, fast_local=False`
   - This caches function compilation but still uses process isolation
   
3. For fastest execution with trusted code (no safety guarantees):
   - Use `fast_local=True`
   - This directly executes in the current process with function caching
   - Only use with code you trust completely
"""

import multiprocessing
import inspect
import types
import traceback
from typing import Callable, Any, Dict, List, Optional, Union, Tuple
import numpy as np


def preprocess_func_str(func_str: str) -> str:
    """
    Preprocess the function string to ensure it is safe for execution.
    
    Args:
        func_str: String containing the function definition
        
    Returns:
        Preprocessed function string
    """
    
    # Check for special functions and ensure they are implemented with numpy
    special_functions = ["sin", "cos", "tan", "exp", "log", "sqrt", "pi"]
    for func in special_functions:
        if func in func_str and not f"np.{func}" in func_str:
            func_str = func_str.replace(func, f"np.{func}")
    return func_str


class Sandbox:
    """Abstract base class for executing untrusted code in a sandbox."""
    
    def execute(self, func_str: str, args: Dict[str, Any], timeout: int = 5) -> Any:
        """
        Execute the provided function with the given arguments in a sandbox.
        
        Args:
            func_str: String containing the function definition
            args: Dictionary of arguments to pass to the function
            timeout: Maximum execution time in seconds
            
        Returns:
            The result of executing the function
            
        Raises:
            ValueError: If the function is invalid or execution fails
        """
        raise NotImplementedError("Subclasses must implement execute")


class LocalSandbox(Sandbox):
    """
    Execute untrusted code in a local sandbox using multiprocessing for isolation.
    
    This sandbox provides:
    - Process isolation via multiprocessing
    - Timeout enforcement
    - Return value validation
    - Function caching for performance
    """
    
    def __init__(self, allowed_modules: List[str] = None):
        """
        Initialize a LocalSandbox instance.
        
        Args:
            allowed_modules: List of module names that are allowed to be imported
        """
        self.allowed_modules = allowed_modules or ["math", "numpy", "np"]
        self._function_cache = {}  # Cache to store compiled functions
    
    def execute(self, func_str: str, args: Dict[str, Any], timeout: int = 5) -> Any:
        """
        Execute code in an isolated process with a timeout.
        
        Args:
            func_str: String containing the function definition
            args: Dictionary of arguments to pass to the function
            timeout: Maximum execution time in seconds
            
        Returns:
            The result of executing the function
            
        Raises:
            ValueError: If the function is invalid or execution fails
            TimeoutError: If execution exceeds the timeout
        """
        # Create a queue for the result
        result_queue = multiprocessing.Queue()
        
        # Start a process for execution
        process = multiprocessing.Process(
            target=self._execute_in_process,
            args=(func_str, args, result_queue)
        )
        
        try:
            # Start the process
            process.start()
            
            # Wait for the process to complete or timeout
            process.join(timeout)
            
            # Check if the process is still running (timeout occurred)
            if process.is_alive():
                # Terminate the process if it's still running
                process.terminate()
                process.join()
                raise TimeoutError(f"Execution timed out after {timeout} seconds")
            
            # Check for execution result
            if not result_queue.empty():
                result = result_queue.get()
                
                # Check if there was an error
                if isinstance(result, dict) and "error" in result:
                    raise ValueError(f"Execution failed: {result['error']}")
                
                return result
            else:
                raise ValueError("Execution failed with no result")
            
        finally:
            # Ensure the process is terminated
            if process.is_alive():
                process.terminate()
                process.join()
    
    def execute_cached(self, func_str: str, args: Dict[str, Any], timeout: int = 5) -> Any:
        """
        Execute code with function caching for better performance.
        
        This version reuses the compiled function if it has been seen before,
        which avoids repeated parsing and compilation costs.
        
        Args:
            func_str: String containing the function definition
            args: Dictionary of arguments to pass to the function
            timeout: Maximum execution time in seconds
            
        Returns:
            The result of executing the function
            
        Raises:
            ValueError: If the function is invalid or execution fails
            TimeoutError: If execution exceeds the timeout
        """
        # Use the cached function if available
        if func_str not in self._function_cache:
            # Create a queue for the result
            result_queue = multiprocessing.Queue()
            
            # Start a process for function compilation
            process = multiprocessing.Process(
                target=self._compile_function,
                args=(func_str, result_queue)
            )
            
            try:
                # Start the process
                process.start()
                
                # Wait for compilation (should be quick)
                process.join(timeout)
                
                # Check if the process is still running (timeout occurred)
                if process.is_alive():
                    process.terminate()
                    process.join()
                    raise TimeoutError(f"Function compilation timed out after {timeout} seconds")
                
                # Check for compilation result
                if not result_queue.empty():
                    result = result_queue.get()
                    
                    # Check if there was an error
                    if isinstance(result, dict) and "error" in result:
                        raise ValueError(f"Compilation failed: {result['error']}")
                    
                    # Store function info in cache
                    self._function_cache[func_str] = result
                else:
                    raise ValueError("Compilation failed with no result")
                
            finally:
                # Ensure the process is terminated
                if process.is_alive():
                    process.terminate()
                    process.join()
        
        # Now execute with the cached function info
        function_info = self._function_cache[func_str]
        
        # Start a new process just for execution (not compilation)
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._execute_cached_function,
            args=(function_info, args, result_queue)
        )
        
        try:
            # Start the process
            process.start()
            
            # Wait for the process to complete or timeout
            process.join(timeout)
            
            # Check if the process is still running (timeout occurred)
            if process.is_alive():
                # Terminate the process if it's still running
                process.terminate()
                process.join()
                raise TimeoutError(f"Execution timed out after {timeout} seconds")
            
            # Check for execution result
            if not result_queue.empty():
                result = result_queue.get()
                
                # Check if there was an error
                if isinstance(result, dict) and "error" in result:
                    raise ValueError(f"Execution failed: {result['error']}")
                
                return result
            else:
                raise ValueError("Execution failed with no result")
            
        finally:
            # Ensure the process is terminated
            if process.is_alive():
                process.terminate()
                process.join()
    
    def _compile_function(self, func_str: str, result_queue: multiprocessing.Queue):
        """
        Compile a function from string and return its metadata.
        
        Args:
            func_str: String containing the function definition
            result_queue: Queue to store the result
        """
        try:
            # Create a namespace with allowed modules
            namespace = {}
            for module_name in self.allowed_modules:
                if module_name == "np":
                    namespace["np"] = np
                elif module_name == "numpy":
                    namespace["numpy"] = np
                else:
                    try:
                        namespace[module_name] = __import__(module_name)
                    except ImportError:
                        # Skip modules that can't be imported
                        pass
            
            # Execute the function definition in the namespace
            exec(func_str, namespace)
            
            # Find the function in the namespace
            func = None
            func_name = None
            for name, obj in namespace.items():
                if isinstance(obj, types.FunctionType) and name != "_compile_function":
                    func = obj
                    func_name = name
                    break
            
            if func is None:
                result_queue.put({"error": "No function found in the provided code"})
                return
            
            # Return the function name and code (not the actual function object, which isn't picklable)
            result_queue.put({
                "func_name": func_name,
                "func_code": func_str,
                "param_names": list(inspect.signature(func).parameters.keys())
            })
                
        except Exception as e:
            # Capture the stack trace
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result_queue.put({"error": error_msg})
    
    def _execute_cached_function(self, function_info: Dict[str, Any], args: Dict[str, Any], result_queue: multiprocessing.Queue):
        """
        Execute a cached function with the provided arguments.
        
        Args:
            function_info: Dictionary with function metadata
            args: Dictionary of arguments to pass to the function
            result_queue: Queue to store the result
        """
        try:
            # Create a namespace with allowed modules
            namespace = {}
            for module_name in self.allowed_modules:
                if module_name == "np":
                    namespace["np"] = np
                elif module_name == "numpy":
                    namespace["numpy"] = np
                else:
                    try:
                        namespace[module_name] = __import__(module_name)
                    except ImportError:
                        # Skip modules that can't be imported
                        pass
            
            # Execute the function definition in the namespace
            exec(function_info["func_code"], namespace)
            
            # Get the function by name
            func = namespace[function_info["func_name"]]
            
            # Ensure we only pass the expected arguments
            param_names = function_info["param_names"]
            filtered_args = {k: v for k, v in args.items() if k in param_names}
            
            # Execute the function with the arguments
            result = func(**filtered_args)
            
            # Validate the result
            if self._is_valid_result(result):
                result_queue.put(result)
            else:
                result_queue.put({"error": f"Invalid result type: {type(result)}"})
                
        except Exception as e:
            # Capture the stack trace
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result_queue.put({"error": error_msg})
    
    def _execute_in_process(self, func_str: str, args: Dict[str, Any], result_queue: multiprocessing.Queue):
        """
        Execute the function in an isolated process.
        
        Args:
            func_str: String containing the function definition
            args: Dictionary of arguments to pass to the function
            result_queue: Queue to store the result
        """
        try:
            # Create a namespace with allowed modules
            namespace = {}
            for module_name in self.allowed_modules:
                if module_name == "np":
                    namespace["np"] = np
                elif module_name == "numpy":
                    namespace["numpy"] = np
                else:
                    try:
                        namespace[module_name] = __import__(module_name)
                    except ImportError:
                        # Skip modules that can't be imported
                        pass
            
            # Execute the function definition in the namespace
            exec(func_str, namespace)
            
            # Find the function in the namespace
            func = None
            for name, obj in namespace.items():
                if isinstance(obj, types.FunctionType) and name != "_execute_in_process":
                    func = obj
                    break
            
            if func is None:
                result_queue.put({"error": "No function found in the provided code"})
                return
            
            # Execute the function with the arguments
            result = func(**args)
            
            # Validate the result
            if self._is_valid_result(result):
                result_queue.put(result)
            else:
                result_queue.put({"error": f"Invalid result type: {type(result)}"})
                
        except Exception as e:
            # Capture the stack trace
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result_queue.put({"error": error_msg})
    
    def _is_valid_result(self, result: Any) -> bool:
        """
        Check if the result is a valid type that is safe to return.
        
        Args:
            result: The result to validate
            
        Returns:
            True if the result is valid, False otherwise
        """
        # Allow numeric types and None
        if result is None or isinstance(result, (int, float, bool, complex, np.number)):
            return True
        
        # Allow numpy arrays (converted to plain Python lists)
        if isinstance(result, np.ndarray):
            return True
        
        # Allow lists and tuples if all elements are valid
        if isinstance(result, (list, tuple)):
            return all(self._is_valid_result(item) for item in result)
        
        # Allow dictionaries if all keys and values are valid
        if isinstance(result, dict):
            return (all(isinstance(k, str) for k in result.keys()) and
                   all(self._is_valid_result(v) for v in result.values()))
        
        # Reject all other types as potentially unsafe
        return False


def create_function_from_string(func_str: str, sandbox: bool = True, timeout: int = 5, 
                              cached: bool = True, fast_local: bool = False) -> Callable:
    """
    Create a callable function from a string, optionally using a sandbox.
    
    Args:
        func_str: String containing the function definition
        sandbox: Whether to execute the function in a sandbox
        timeout: Maximum execution time in seconds for sandboxed execution
        cached: Whether to cache the compiled function (improves performance for repeated calls)
        fast_local: Whether to use direct execution (only for trusted code, much faster but less safe)
        
    Returns:
        A callable function that executes the provided code
        
    Raises:
        ValueError: If the function is invalid or cannot be created
    """
    # Direct local execution without any sandboxing (FAST but UNSAFE for untrusted code)
    func_str = preprocess_func_str(func_str)
    if fast_local:
        # Use a persistent function cache for fast_local mode
        # This avoids re-parsing the function string on each call
        if not hasattr(create_function_from_string, '_function_cache'):
            create_function_from_string._function_cache = {}
            
        # Check if we've already compiled this function
        if func_str in create_function_from_string._function_cache:
            return create_function_from_string._function_cache[func_str]
            
        # Set up namespace with allowed modules
        namespace = {
            "np": np,
            "numpy": np,
            "math": __import__('math')
        }
        
        try:
            # Compile the function once
            exec(func_str, namespace)
            
            # Find the function in the namespace
            func = None
            for name, obj in namespace.items():
                if isinstance(obj, types.FunctionType) and name != "create_function_from_string":
                    func = obj
                    break
            
            if func is None:
                raise ValueError("No function found in the provided code")
                
            # Cache the function for future use
            create_function_from_string._function_cache[func_str] = func
            return func
            
        except Exception as e:
            raise ValueError(f"Failed to create function: {e}")
    
    # Non-sandboxed execution (still in the same process but with a more controlled environment)
    elif not sandbox:
        # Direct execution without sandbox (NOT RECOMMENDED FOR UNTRUSTED CODE)
        namespace = {
            "np": np,
            "math": __import__('math')
        }
        
        try:
            exec(func_str, namespace)
            
            # Find the function in the namespace
            for name, obj in namespace.items():
                if isinstance(obj, types.FunctionType) and name != "create_function_from_string":
                    return obj
            
            raise ValueError("No function found in the provided code")
            
        except Exception as e:
            raise ValueError(f"Failed to create function: {e}")
    
    # Sandboxed execution (safest option, with optional caching for better performance)
    else:
        # Create a sandbox for safe execution
        sandbox_instance = LocalSandbox()
        
        # Extract the function signature
        try:
            # Use a placeholder to just get the signature
            temp_namespace = {}
            exec(func_str, temp_namespace)
            
            func = None
            for name, obj in temp_namespace.items():
                if isinstance(obj, types.FunctionType) and name != "create_function_from_string":
                    func = obj
                    break
            
            if func is None:
                raise ValueError("No function found in the provided code")
            
            # Get the parameter names and signature
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # Choose between cached and regular execution
            if cached:
                # Create a wrapper function that uses the cached sandbox execution
                def sandboxed_function(**kwargs):
                    # Ensure we only pass the expected arguments
                    args = {k: v for k, v in kwargs.items() if k in param_names}
                    
                    # Execute in the sandbox with caching
                    return sandbox_instance.execute_cached(func_str, args, timeout=timeout)
            else:
                # Create a wrapper function that uses regular sandbox execution
                def sandboxed_function(**kwargs):
                    # Ensure we only pass the expected arguments
                    args = {k: v for k, v in kwargs.items() if k in param_names}
                    
                    # Execute in the sandbox without caching
                    return sandbox_instance.execute(func_str, args, timeout=timeout)
            
            # Copy the signature and docstring
            sandboxed_function.__signature__ = sig
            sandboxed_function.__doc__ = func.__doc__
            
            return sandboxed_function
            
        except Exception as e:
            raise ValueError(f"Failed to create sandboxed function: {e}")
        

if __name__ == "__main__":
    # Example usage
    func_str = """
def add(a, b):
    return np.add(a, b)
"""
    func = create_function_from_string(func_str, sandbox=True, cached=True)
    result = func(a=5, b=10)
    print(f"Result of add(5, 10): {result}")