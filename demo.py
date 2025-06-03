import time
from typing import Dict, Any, List
from traitDict import traits
from cognitiveState import CognitiveState, MemoryType
from goalSpace import GoalSpace, Goal, GoalStatus
from scheduler import GoalScheduler
from goalsEngine import GoalEngine
from arbitration import ArbitrationEngine, ArbitrationConfig, ArbitrationType
import numpy as np

def print_trait_info(engine: GoalEngine):
    """Print initial trait information"""
    print("\nInitial Traits:")
    print("==============")
    for name, data in engine.traits.items():
        print(f"{name}: {data['value']:.3f}")

def print_system_status(scheduler):
    """Print detailed system status including traits, goals, and metrics"""
    print("\n=== System Status ===")
    
    # Print cognitive state
    print("\nCognitive State:")
    print(f"Memory: {scheduler.cognitive_state.memory}")
    print(f"Traits: {scheduler.cognitive_state.traits}")
    print(f"Affective State: valence={scheduler.cognitive_state.affective.valence:.2f}, "
          f"arousal={scheduler.cognitive_state.affective.arousal:.2f}")
    
    # Print goal space
    print("\nGoal Space:")
    print(f"Total Goals: {len(scheduler.goal_space.goals)}")
    for name, goal in scheduler.goal_space.goals.items():
        print(f"\n{name}:")
        print(f"Description: {goal.attributes.get('description', '')}")
        print(f"Progress: {goal.progress:.2f}")
        print(f"Utility: {scheduler.goal_space.get_utility(name):.3f}")
        print(f"Weight: {scheduler.goal_space.arbitration_weights[name]:.3f}")
        if goal.attributes.get('linked_traits'):
            print(f"Linked Traits: {', '.join(goal.attributes['linked_traits'])}")
    
    # Print stability metrics
    metrics = scheduler.goal_space.arbitration_engine.get_stability_metrics()
    print("\nStability Metrics:")
    print(f"Max difference: {metrics['max_diff']:.6f}")
    print(f"Mean difference: {metrics['mean_diff']:.6f}")
    print(f"Convergence rate: {metrics['convergence_rate']:.6f}")
    
    # Print Lipschitz bound
    print(f"\nLipschitz bound: {scheduler.goal_space.arbitration_engine.get_lipschitz_bound():.6f}")
    
    # Print conflicts if any
    conflicts = scheduler.goal_space.detect_conflicts()
    if conflicts:
        print("\nGoal Conflicts:")
        for g1, g2, resolution in conflicts:
            print(f"{g1} vs {g2}: {resolution}")

def create_recursive_goals():
    """Create goals with recursive dependencies"""
    # Define recursive utility functions
    def devotion_utility(dep_utils, weights):
        # Devotion depends on progress of other goals
        return 0.7 + 0.3 * sum(dep_utils.values())
    
    def learning_utility(dep_utils, weights):
        # Learning depends on devotion and exploration
        return 0.5 + 0.3 * dep_utils.get('devotion_goal', 0.0) + 0.2 * dep_utils.get('exploration_goal', 0.0)
    
    def exploration_utility(dep_utils, weights):
        # Exploration depends on learning and devotion
        return 0.6 + 0.2 * dep_utils.get('learning_goal', 0.0) + 0.2 * dep_utils.get('devotion_goal', 0.0)
    
    # Create goals with recursive dependencies
    # Note: Dependencies form a DAG, but utilities can still be recursive
    goals = [
        Goal(
            name='devotion_goal',
            attributes={'commitment': 0.8, 'consistency': 0.7},
            priority=0.9,
            dependencies=set(),  # Root goal
            utility_fn=devotion_utility
        ),
        Goal(
            name='learning_goal',
            attributes={'curiosity': 0.8, 'adaptability': 0.6},
            priority=0.8,
            dependencies={'devotion_goal'},  # Depends on devotion
            utility_fn=learning_utility
        ),
        Goal(
            name='exploration_goal',
            attributes={'creativity': 0.7, 'risk_tolerance': 0.5},
            priority=0.7,
            dependencies={'learning_goal'},  # Depends on learning
            utility_fn=exploration_utility
        )
    ]
    
    return goals

def main():
    # Create scheduler with default settings
    scheduler = GoalScheduler(
        tick_interval=1.0,    # 1 second between ticks
        progress_delta=0.1,   # Progress goals by 10% each tick
        decay_rate=0.05       # 5% trait decay per tick
    )
    
    print("Initializing Goal Scheduler...")
    print("\nInitial Traits:")
    for name, data in scheduler.engine.traits.items():
        print(f"\n{name}:")
        print(f"Value: {data['value']:.2f}")
        print(f"Description: {data['description']}")
        if data.get('links'):
            print(f"Linked traits: {', '.join(data['links'])}")
    
    print("\nStarting perpetual goal evaluation...")
    try:
        while True:
            # Run one tick of the scheduler
            scheduler.ticks += 1
            print(f"\n=== Tick {scheduler.ticks} ===")
            
            # Update engine and cognitive state
            scheduler.engine.update_goals()
            scheduler._generate_goals_from_traits()
            scheduler._progress_goals()
            scheduler._resolve_satisfied_goals()
            scheduler._feedback_to_traits()
            
            # Print detailed system status
            print_system_status(scheduler)
            
            # Print number of completed goals
            print(f"\nGoals completed so far: {scheduler.completed_goals}")
            
            # Sleep for tick interval
            time.sleep(scheduler.tick_interval)
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
        print(f"Final tick count: {scheduler.ticks}")
        print(f"Total goals completed: {scheduler.completed_goals}")

if __name__ == "__main__":
    main()
