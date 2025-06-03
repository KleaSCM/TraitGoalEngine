import time
from typing import Dict, Any, List
from traitDict import traits
from cognitiveState import CognitiveState, MemoryType
from goalSpace import GoalSpace, Goal, GoalStatus
from scheduler import GoalScheduler
from goalsEngine import GoalEngine
from arbitration import ArbitrationEngine, ArbitrationConfig, ArbitrationType
import numpy as np
from dynamicUpdates import UpdateType
import random

def print_trait_info(engine: GoalEngine):
    """Print initial trait information"""
    print("\nInitial Traits:")
    print("==============")
    for name, data in engine.traits.items():
        print(f"{name}: {data['value']:.3f}")

def print_system_status(goal_space):
    """Print detailed system status including dynamic metrics."""
    print("\n=== System Status ===")
    
    # Print goal space
    print("\nGoal Space:")
    for goal in goal_space.goals.values():
        print(f"\n  {goal.name}:")
        print(f"    Description: {goal.description}")
        print(f"    Progress: {goal.progress:.2%}")
        print(f"    Status: {goal.status.value}")
        print(f"    Linked Traits: {goal.linked_traits}")
        print(f"    Dependencies: {goal.dependencies}")
    
    # Print utilities and weights
    utilities = goal_space.evaluate_all_utilities()
    weights = goal_space.arbitrate_goals()
    
    print("\nUtilities and Weights:")
    for goal_name in goal_space.goals:
        print(f"  {goal_name}:")
        print(f"    Utility: {utilities[goal_name]:.4f}")
        print(f"    Weight: {weights[goal_name]:.4f}")
        print(f"    Progress Rate: {(weights[goal_name] * utilities[goal_name] * 0.5):.4f} per tick")
    
    # Print stability metrics
    metrics = goal_space.get_stability_metrics()
    print("\nStability Metrics:")
    print(f"  Max Difference: {metrics['max_diff']:.6f}")
    print(f"  Mean Difference: {metrics['mean_diff']:.6f}")
    print(f"  Std Difference: {metrics['std_diff']:.6f}")
    print(f"  Convergence Rate: {metrics['convergence_rate']:.6f}")
    
    # Print Lipschitz bound
    lipschitz = goal_space.get_lipschitz_bound()
    print(f"\nLipschitz Bound: {lipschitz:.6f}")
    
    # Print conflicts
    conflicts = goal_space.detect_conflicts()
    if conflicts:
        print("\nPotential Goal Conflicts:")
        for g1, g2 in conflicts:
            print(f"  {g1} ↔ {g2}")

def generate_goal_from_traits(trait_name: str, trait_data: Dict[str, Any]) -> Goal:
    """
    Generate a goal based on a trait's current state and properties.
    """
    # Create utility function that depends on linked traits
    def trait_utility(dep_utilities):
        # Base utility from trait value
        base_utility = trait_data.get('value', 0.5)
        
        # Add influence from linked traits
        linked_influence = 0.0
        for linked_trait in trait_data.get('links', []):
            linked_utility = dep_utilities.get(f"{linked_trait}_goal", 0.0)
            linked_influence += 0.3 * linked_utility  # Coupling coefficient
        
        # Add some randomness to break symmetry
        random_factor = 0.1 * (random.random() - 0.5)
        return base_utility + linked_influence + random_factor
    
    # Generate goal attributes based on trait properties
    attributes = {
        'value': trait_data.get('value', 0.5),
        'stability': trait_data.get('stability', 0.5),
        'valence': trait_data.get('valence', 0.0)
    }
    
    # Create goal with trait-based properties
    return Goal(
        name=f"{trait_name}_goal",
        attributes=attributes,
        priority=trait_data.get('value', 0.5),  # Priority based on trait value
        dependencies=set(trait_data.get('links', [])),  # Dependencies from trait links
        utility_fn=trait_utility,
        description=f"Develop and strengthen {trait_name} through focused practice and reflection",
        linked_traits=[trait_name] + trait_data.get('links', []),
        status=GoalStatus.ACTIVE  # Set initial status to ACTIVE
    )

def should_generate_goal(trait_name: str, trait_data: Dict[str, Any], goal_space: GoalSpace) -> bool:
    """
    Determine if a goal should be generated for this trait based on its state.
    """
    # Don't generate if goal already exists
    goal_name = f"{trait_name}_goal"
    if goal_name in goal_space.goals:
        return False
    
    # Generate goal if trait value is below threshold or has strong links
    trait_value = trait_data.get('value', 0.5)
    has_links = len(trait_data.get('links', [])) > 0
    
    return trait_value < 0.8 or has_links

def main():
    # Initialize goal space with dynamic updates
    goal_space = GoalSpace(
        update_type=UpdateType.CONTINUOUS,  # Use continuous-time dynamics
        arbitration_rule="softmax",
        temperature=1.0
    )
    
    # Track completed goals
    completed_goals = set()
    
    # Run perpetual goal evaluation
    print("Starting perpetual goal evaluation...")
    print("Press Ctrl+C to stop")
    
    tick = 0
    try:
        while True:
            print(f"\nTick {tick}")
            
            # Generate new goals from traits
            for trait_name, trait_data in traits.items():
                if should_generate_goal(trait_name, trait_data, goal_space):
                    try:
                        goal = generate_goal_from_traits(trait_name, trait_data)
                        goal_space.add_goal(goal)
                        print(f"\nGenerated new goal: {goal.name}")
                        print(f"  Description: {goal.description}")
                        print(f"  Dependencies: {goal.dependencies}")
                    except ValueError as e:
                        # Skip if adding would create a cycle
                        print(f"Skipping {trait_name}_goal: {str(e)}")
            
            # Update goal progress
            for goal in list(goal_space.goals.values()):  # Use list to avoid modification during iteration
                if goal.status == GoalStatus.ACTIVE:
                    # Simulate progress based on current weights and utility
                    weights = goal_space.arbitrate_goals()
                    utilities = goal_space.evaluate_all_utilities()
                    
                    # Progress depends on both weight and utility, with some randomness
                    base_progress = weights[goal.name] * utilities[goal.name]
                    random_factor = 0.1 * (random.random() - 0.5)  # Add some randomness
                    progress_delta = (base_progress + random_factor) * 0.5  # Increased from 0.2 to 0.5
                    
                    goal_space.update_goal_progress(goal.name, progress_delta)
                    
                    # Check for goal completion
                    if goal.progress >= 1.0:  # Changed from checking status to checking progress
                        goal.status = GoalStatus.SATISFIED
                        completed_goals.add(goal.name)
                        print(f"\nGoal completed: {goal.name}")
                        print(f"  Description: {goal.description}")
                        print(f"  Final progress: {goal.progress:.2f}")
                        
                        # Remove completed goal from goal space and update arbitration
                        goal_space.goals.pop(goal.name)
                        if goal.name in goal_space.dependencies:
                            goal_space.dependencies.pop(goal.name)
                        
                        # Update weights for remaining goals
                        goal_space.arbitration_engine.update_weights(utilities)
            
            # Print system status
            print_system_status(goal_space)
            
            # Print completion statistics
            print(f"\nCompletion Statistics:")
            print(f"  Total goals completed: {len(completed_goals)}")
            print(f"  Active goals: {len(goal_space.goals)}")
            if len(completed_goals) + len(goal_space.goals) > 0:
                print(f"  Completion rate: {len(completed_goals) / (len(completed_goals) + len(goal_space.goals)):.2%}")
            
            # Increment tick counter
            tick += 1
            
            # Small delay to make output readable
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping goal evaluation...")
        print(f"\nFinal Statistics:")
        print(f"  Total ticks: {tick}")
        print(f"  Total goals completed: {len(completed_goals)}")
        if len(completed_goals) + len(goal_space.goals) > 0:
            print(f"  Completion rate: {len(completed_goals) / (len(completed_goals) + len(goal_space.goals)):.2%}")
        print("\nCompleted goals:")
        for goal_name in completed_goals:
            print(f"  - {goal_name}")

if __name__ == "__main__":
    main()
