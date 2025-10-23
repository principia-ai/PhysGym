#!/usr/bin/env python3
"""
Basic usage example for PhysGym package.

This script demonstrates how to use PhysGym after installing it as a package.
"""

import physgym
import numpy as np

def main():
    print("="*60)
    print("PhysGym Basic Usage Example")
    print("="*60)
    
    # 1. Create a physics environment directly
    print("\n1. Loading a physics environment...")
    env = physgym.PhyEnv(285)
    print(f"   Environment ID: {env.id}")
    print(f"   Problem: {env.problem_content[:80]}...")
    print(f"   Input variables: {list(env.input_variables_des.keys())}")
    print(f"   Output variable: {list(env.output_variable_des.keys())}")
    
    # 2. Create a research interface
    print("\n2. Creating a research interface...")
    experiment = physgym.ResearchInterface(
        env=285, 
        sample_quota=50,
        test_quota=2,
        mode="default"
    )
    print(f"   Sample quota: {experiment.get_remaining_quota()}")
    print(f"   Controllable variables: {list(experiment.controllable_variables.keys())}")
    
    # 3. Generate and run some experiments
    print("\n3. Running sample experiments...")
    input_samples = []
    for i in range(5):
        sample = {}
        for param in experiment.input_params:
            sample[param] = np.random.uniform(0.1, 10.0)
        input_samples.append(sample)
    
    results = experiment.run_experiment(input_samples)
    print(f"   Ran {len(results)} experiments")
    print(f"   Remaining quota: {experiment.get_remaining_quota()}")
    print(f"   Sample result: {results[0]}")
    
    # 4. Test a hypothesis
    print("\n4. Testing a hypothesis...")
    hypothesis_code = """
def hypothesis_function(m, R, omega):
    '''Simple hypothesis: linear relationship'''
    return m * R * omega
"""
    hypothesis_expr = "m * R * omega"
    
    evaluation = experiment.test_hypothesis(
        candidate_function=hypothesis_code,
        candidate_expr=hypothesis_expr
    )
    
    if evaluation and "error" not in evaluation:
        print(f"   Hypothesis evaluation completed")
        if 'mse' in evaluation:
            print(f"   MSE: {evaluation['mse']:.4f}")
        if 'r2' in evaluation:
            print(f"   R²: {evaluation['r2']:.4f}")
    else:
        print(f"   Hypothesis evaluation failed: {evaluation}")
    
    # 5. Check experiment status
    print("\n5. Checking experiment status...")
    print(f"   Total observations: {len(experiment.get_observations())}")
    print(f"   Samples used: {experiment.samples_used}")
    print(f"   Remaining quota: {experiment.get_remaining_quota()}")
    
    print("\n" + "="*60)
    print("✓ Example completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()

