"""
Test script for sandbox performance improvements.
"""

import time
from sandbox import create_function_from_string
import numpy as np

# A simple function to test
test_function = """
def test_function(x, y):
    return x + y
"""

# Function with numpy operation
numpy_function = """
def numpy_operation(x, y):
    a = np.array(x)
    b = np.array(y)
    return np.dot(a, b)
"""

def time_execution(func, args, runs=10):
    """Measure execution time for a function."""
    start_time = time.time()
    for _ in range(runs):
        result = func(**args)
    end_time = time.time()
    return (end_time - start_time) / runs, result

def main():
    print("Testing sandbox performance improvements")
    print("-" * 50)
    
    # Test parameters
    args = {'x': 3, 'y': 4}
    numpy_args = {'x': [1, 2, 3], 'y': [4, 5, 6]}
    runs = 100
    
    # 1. Standard sandbox (slowest but safest)
    print("\n1. Standard sandbox (new process per call):")
    standard_func = create_function_from_string(test_function, sandbox=True, cached=False, fast_local=False)
    time_taken, result = time_execution(standard_func, args, runs)
    print(f"   Average time: {time_taken:.6f} seconds")
    print(f"   Result: {result}")
    
    # 2. Cached sandbox (better performance with safety)
    print("\n2. Cached sandbox (process per call with cached compilation):")
    cached_func = create_function_from_string(test_function, sandbox=True, cached=True, fast_local=False)
    time_taken, result = time_execution(cached_func, args, runs)
    print(f"   Average time: {time_taken:.6f} seconds")
    print(f"   Result: {result}")
    
    # 3. Fast local execution (fastest, no sandbox)
    print("\n3. Fast local execution (no sandbox, direct execution):")
    fast_func = create_function_from_string(test_function, fast_local=True)
    time_taken, result = time_execution(fast_func, args, runs)
    print(f"   Average time: {time_taken:.6f} seconds")
    print(f"   Result: {result}")
    
    # Test with NumPy operations
    print("\n\nTesting with NumPy operations:")
    print("-" * 50)
    
    # 1. Standard sandbox with numpy
    print("\n1. Standard sandbox with NumPy:")
    standard_np_func = create_function_from_string(numpy_function, sandbox=True, cached=False, fast_local=False)
    time_taken, result = time_execution(standard_np_func, numpy_args, runs)
    print(f"   Average time: {time_taken:.6f} seconds")
    print(f"   Result: {result}")
    
    # 2. Cached sandbox with numpy
    print("\n2. Cached sandbox with NumPy:")
    cached_np_func = create_function_from_string(numpy_function, sandbox=True, cached=True, fast_local=False)
    time_taken, result = time_execution(cached_np_func, numpy_args, runs)
    print(f"   Average time: {time_taken:.6f} seconds")
    print(f"   Result: {result}")
    
    # 3. Fast local with numpy
    print("\n3. Fast local execution with NumPy:")
    fast_np_func = create_function_from_string(numpy_function, fast_local=True)
    time_taken, result = time_execution(fast_np_func, numpy_args, runs)
    print(f"   Average time: {time_taken:.6f} seconds")
    print(f"   Result: {result}")
    
    # Summary
    print("\n\nPerformance Summary:")
    print("-" * 50)
    print("For best performance with trusted code: use fast_local=True")
    print("For a balance of safety and performance: use sandbox=True, cached=True")
    print("For maximum isolation (slowest): use sandbox=True, cached=False")

if __name__ == "__main__":
    main()