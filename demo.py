import time
from typing import Dict, Any, List
from traitDict import traits
from cognitiveState import CognitiveState, MemoryType
from goalSpace import GoalSpace, Goal, GoalStatus, GoalSpaceConfig
from scheduler import GoalScheduler
from goalsEngine import GoalEngine
from arbitration import ArbitrationEngine, ArbitrationConfig, ArbitrationType, NashConfig
import numpy as np
from dynamicUpdates import UpdateType
import random
from traitGoalEngine import calculate_goal_priorities, prepare_traits_for_goals_engine, update_trait_values
from stochasticDynamics import SDEConfig

def print_trait_info(engine: GoalEngine):
    """Print initial trait information"""
    print("\nInitial Traits:")
    print("==============")
    for name, data in engine.traits.items():
        print(f"{name}: {data['value']:.3f}")

def print_system_status(goal_space, trait_values: Dict[str, float]):
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
    utilities = goal_space.evaluate_all_utilities(trait_values)
    weights = goal_space.arbitrate_goals(trait_values)
    
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

def generate_goal_from_traits(traits: Dict[str, Dict[str, Any]], goal_id: int) -> Goal:
    """Generate a goal with trait-based attributes."""
    # Select random traits for this goal
    trait_names = list(traits.keys())
    num_traits = min(3, len(trait_names))
    selected_traits = np.random.choice(trait_names, num_traits, replace=False)
    
    # Create goal with trait-based attributes
    goal = Goal(
        name=f"Goal_{goal_id}",
        attributes={
            'priority': np.random.uniform(0.3, 0.9),
            'stability': np.random.uniform(0.2, 0.8),
            'resilience': np.random.uniform(0.2, 0.8),
            'decisiveness': np.random.uniform(0.2, 0.8)
        },
        priority=np.random.uniform(0.3, 0.9),
        dependencies=set(),
        linked_traits={trait: np.random.uniform(0.2, 0.8) for trait in selected_traits}
    )
    
    # Add utility function that considers traits
    def utility_fn(trait_values: Dict[str, float]) -> float:
        base_utility = goal.attributes['priority']
        trait_influence = sum(
            goal.linked_traits[trait] * trait_values.get(trait, 0.5)
            for trait in goal.linked_traits
        )
        return base_utility * (1 + trait_influence)
    
    goal.utility_fn = utility_fn
    return goal

def should_generate_goal(goal_space: GoalSpace, traits: Dict[str, Dict[str, Any]]) -> bool:
    """Determine if a new goal should be generated based on current state."""
    if len(goal_space.goals) < 3:  # Always have at least 3 goals
        return True
    
    # Check if any goals are completed
    completed = sum(1 for g in goal_space.goals.values() if g.completed)
    if completed > 0:
        return True
    
    # Check stability metrics
    metrics = goal_space.get_stability_metrics()
    if metrics.get('utility_stability', 0) < 0.3:  # Low stability suggests need for new goals
        return True
    
    return False

def main():
    # Initialize configuration
    sde_config = SDEConfig(
        time_step=0.01,
        noise_scale=0.1,
        mean_reversion_rate=0.1,
        stability_factor=1.0,
        base_temperature=1.0,
        resilience_weight=0.5,
        decisiveness_weight=0.5
    )
    
    goal_space_config = GoalSpaceConfig(
        temperature=1.0,
        stability_threshold=0.1,
        progress_threshold=0.8,
        sde_config=sde_config
    )
    
    # Initialize goal space
    goal_space = GoalSpace(config=goal_space_config)
    
    # Get traits
    traits = prepare_traits_for_goals_engine()
    
    # Track completion statistics
    completed_goals = set()
    completion_times = {}
    
    # Main simulation loop
    max_steps = 100
    for step in range(max_steps):
        print(f"\nStep {step + 1}/{max_steps}")
        
        # Generate new goals if needed
        if should_generate_goal(goal_space, traits):
            new_goal = generate_goal_from_traits(traits, len(goal_space.goals) + 1)
            goal_space.add_goal(new_goal)
            print(f"Generated new goal: {new_goal.name}")
            print(f"Linked traits: {new_goal.linked_traits}")
        
        # Update trait values
        trait_values = {name: data['value'] for name, data in traits.items()}
        
        # Get current utilities and weights
        utilities = goal_space.evaluate_all_utilities(trait_values)
        weights = goal_space.arbitrate_goals(trait_values)
        
        # Print current state
        print("\nCurrent State:")
        for goal_name, goal in goal_space.goals.items():
            print(f"{goal_name}:")
            print(f"  Progress: {goal.progress:.2f}")
            print(f"  Utility: {utilities[goal_name]:.2f}")
            print(f"  Weight: {weights[goal_name]:.2f}")
            print(f"  Traits: {goal.linked_traits}")
            if goal.completed and goal_name not in completed_goals:
                completed_goals.add(goal_name)
                completion_times[goal_name] = step + 1
                print(f"  ✓ COMPLETED at step {step + 1}")
        
        # Check for conflicts
        conflict_groups = goal_space.arbitration_engine.detect_conflicts()
        if conflict_groups:
            print("\nDetected Conflicts:")
            for i, group in enumerate(conflict_groups, 1):
                print(f"Conflict Group {i}: {group}")
        
        # Get stability metrics
        stability_metrics = goal_space.get_stability_metrics()
        print("\nStability Metrics:")
        for metric, value in stability_metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        # Get conflict metrics
        conflict_metrics = goal_space.arbitration_engine.get_conflict_metrics()
        print("\nConflict Metrics:")
        for metric, value in conflict_metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        # Print completion statistics
        active_goals = len(goal_space.goals) - len(completed_goals)
        completion_rate = len(completed_goals) / len(goal_space.goals) if goal_space.goals else 0.0
        print("\nCompletion Statistics:")
        print(f"  Total Goals: {len(goal_space.goals)}")
        print(f"  Completed Goals: {len(completed_goals)}")
        print(f"  Active Goals: {active_goals}")
        print(f"  Completion Rate: {completion_rate:.2%}")
        if completed_goals:
            print("\nCompleted Goals:")
            for goal_name in sorted(completed_goals):
                print(f"  {goal_name} (Step {completion_times[goal_name]})")
        
        # Update goal progress
        for goal_name, goal in goal_space.goals.items():
            if not goal.completed:
                # Calculate progress delta based on weights and traits
                base_progress = weights[goal_name] * 0.1
                trait_influence = sum(
                    goal.linked_traits[trait] * trait_values.get(trait, 0.5)
                    for trait in goal.linked_traits
                )
                progress_delta = base_progress * (1 + trait_influence)
                
                # Add some randomness scaled by stability
                stability = goal.attributes.get('stability', 0.5)
                noise = np.random.normal(0, 0.05 * (1 - stability))
                progress_delta += noise
                
                # Update progress
                goal_space.update_goal_progress(goal_name, progress_delta, trait_values)
        
        # Update trait values based on goal progress
        for name, data in traits.items():
            # Update trait value based on goal progress
            goal_influence = sum(
                goal.progress * goal.linked_traits.get(name, 0)
                for goal in goal_space.goals.values()
            )
            data['value'] = max(0.0, min(1.0, data['value'] + 0.1 * goal_influence))

if __name__ == "__main__":
    main()
