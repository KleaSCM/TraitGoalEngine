from recursiveGoals import RecursiveGoalSystem, RecursiveGoalConfig
import numpy as np

def main():
    # Create recursive goal system
    config = RecursiveGoalConfig(
        max_iterations=100,
        convergence_threshold=1e-6,
        utility_contraction_bound=0.9,
        arbitration_temperature=1.0
    )
    system = RecursiveGoalSystem(config)
    
    # Add goals with recursive dependencies
    # Example from Section 6.3:
    # U1 = 1/2 x1 + 1/2 U2
    # U2 = 1/3 x2 + 2/3 U1
    
    def goal1_utility(dep_utils, weights):
        x1 = 0.7  # Fixed input
        return 0.5 * x1 + 0.5 * dep_utils.get('goal2', 0.0)
    
    def goal2_utility(dep_utils, weights):
        x2 = 0.5  # Fixed input
        return 0.33 * x2 + 0.67 * dep_utils.get('goal1', 0.0)
    
    # Add goals with their dependencies
    system.add_goal('goal1', goal1_utility, {'goal2'})
    system.add_goal('goal2', goal2_utility, {'goal1'})
    
    print("Initial State:")
    print(f"Utilities: {system.utilities}")
    print(f"Weights: {system.arbitration_weights}")
    print("\nRunning recursive evaluation...")
    
    # Run system for several steps
    for step in range(10):
        utilities, weights = system.step({})
        print(f"\nStep {step + 1}:")
        print(f"Utilities: {utilities}")
        print(f"Weights: {weights}")
        
        # Print stability metrics
        if step > 0:
            metrics = system.get_stability_metrics()
            bounds = system.get_lipschitz_bounds()
            print("\nStability Metrics:")
            print(f"Max Utility Difference: {metrics['max_util_diff']:.6f}")
            print(f"Mean Utility Difference: {metrics['mean_util_diff']:.6f}")
            print(f"Utility Convergence Rate: {metrics['util_convergence_rate']:.6f}")
            print(f"Weight Convergence Rate: {metrics['weight_convergence_rate']:.6f}")
            print("\nLipschitz Bounds:")
            print(f"Utility Lipschitz: {bounds['utility_lipschitz']:.6f}")
            print(f"Arbitration Lipschitz: {bounds['arbitration_lipschitz']:.6f}")

if __name__ == "__main__":
    main() 