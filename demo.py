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
from traitGoalEngine import calculate_goal_priorities, prepare_traits_for_goals_engine, update_trait_values

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

def generate_goal_from_traits(trait_name: str, trait_data: Dict[str, Any]) -> Goal:
    """
    Generate a goal based on a trait's current state and properties.
    """
    # Get trait parameters for temperature modulation
    resilience = trait_data.get('resilience', 0.5)
    decisiveness = trait_data.get('decisiveness', 0.5)
    effective_temp = 1.0 / (resilience * decisiveness)  # Temperature modulation
    
    # Create utility function that depends on linked traits
    def trait_utility(dep_utilities):
        # Base utility from trait value
        base_utility = trait_data.get('value', 0.5)
        
        # Add influence from linked traits
        linked_influence = 0.0
        for linked_trait in trait_data.get('links', []):
            linked_utility = dep_utilities.get(f"{linked_trait}_goal", 0.0)
            linked_influence += 0.3 * linked_utility  # Coupling coefficient
        
        # Mean-reverting utility process
        mean_utility = base_utility + linked_influence
        reversion_rate = 0.1  # λ parameter
        current_utility = getattr(trait_utility, 'last_utility', mean_utility)
        new_utility = mean_utility + reversion_rate * (current_utility - mean_utility)
        
        # Add noise proportional to temperature
        noise = np.random.normal(0, 0.1 * effective_temp)
        trait_utility.last_utility = new_utility + noise
        
        return trait_utility.last_utility
    
    # Generate goal attributes based on trait properties
    attributes = {
        'value': trait_data.get('value', 0.5),
        'stability': trait_data.get('stability', 0.5),
        'valence': trait_data.get('valence', 0.0),
        'resilience': resilience,
        'decisiveness': decisiveness,
        'temperature': effective_temp
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
    print("Starting perpetual goal evaluation...")
    print("Press Ctrl+C to stop\n")
    
    # Initialize goal space with continuous-time dynamics
    goal_space = GoalSpace(
        update_type=UpdateType.CONTINUOUS,
        arbitration_rule="softmax",
        temperature=1.0
    )
    
    # Initialize trait engine
    trait_engine = GoalEngine()
    trait_engine.traits = prepare_traits_for_goals_engine()
    
    tick = 0
    completed_goals = set()
    
    try:
        while True:
            print(f"\nTick {tick}\n")
            
            # Update traits and generate goals
            priorities, updated_traits = calculate_goal_priorities()
            trait_engine.traits = updated_traits
            
            # Generate goals from traits
            for trait_name, trait_data in trait_engine.traits.items():
                if should_generate_goal(trait_name, trait_data, goal_space):
                    goal = generate_goal_from_traits(trait_name, trait_data)
                    try:
                        goal_space.add_goal(goal)
                        print(f"\nGenerated new goal: {goal.name}")
                        print(f"  Description: {goal.description}")
                        print(f"  Dependencies: {goal.dependencies}")
                        print(f"  Temperature: {goal_space.get_effective_temperature():.3f}")
                    except ValueError as e:
                        print(f"Could not add goal {goal.name}: {e}")
            
            # Calculate weights and utilities after all goals are added
            if goal_space.goals:  # Only proceed if there are goals
                # Get trait values from trait engine
                trait_values = {name: data['value'] for name, data in trait_engine.traits.items()}
                
                # Calculate weights and utilities
                weights = goal_space.arbitrate_goals(trait_values)
                utilities = goal_space.evaluate_all_utilities(trait_values)
                
                # Print detailed system status
                print_system_status(goal_space, trait_values)
                
                # Update progress for each active goal
                for goal in goal_space.goals.values():
                    if goal.status == GoalStatus.ACTIVE:
                        # Calculate progress based on weights and utilities
                        base_progress = weights[goal.name] * utilities[goal.name]
                        random_factor = 0.1 * (random.random() - 0.5)  # Add some randomness
                        progress_delta = (base_progress + random_factor) * 0.2
                        
                        goal_space.update_goal_progress(goal.name, progress_delta)
                        
                        # Check for completion
                        if goal.progress >= 0.8:  # Changed from 1.0 to 0.8 to match satisfaction threshold
                            goal.status = GoalStatus.SATISFIED
                            completed_goals.add(goal.name)
                            print(f"\nGoal completed: {goal.name}")
                            print(f"  Final progress: {goal.progress:.2f}")
                            print(f"  Final utility: {utilities[goal.name]:.3f}")
                
                # Print completion statistics
                active_goals = sum(1 for g in goal_space.goals.values() if g.status == GoalStatus.ACTIVE)
                completion_rate = len(completed_goals) / (len(completed_goals) + active_goals) if (len(completed_goals) + active_goals) > 0 else 0.0
                
                print(f"\nCompletion Statistics:")
                print(f"  Total completed: {len(completed_goals)}")
                print(f"  Active goals: {active_goals}")
                print(f"  Completion rate: {completion_rate:.2%}")
            
            tick += 1
            time.sleep(1)  # Add a small delay between ticks
            
    except KeyboardInterrupt:
        print("\n\nFinal Statistics:")
        print(f"Total ticks: {tick}")
        print(f"Total goals completed: {len(completed_goals)}")
        print("\nCompleted goals:")
        for goal_name in completed_goals:
            print(f"- {goal_name}")

if __name__ == "__main__":
    main()
