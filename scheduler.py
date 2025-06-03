import time
from typing import List, Optional, Dict, Any
from cognitiveState import CognitiveState
from goalSpace import GoalSpace, Goal, GoalStatus
from goalsEngine import GoalEngine
from arbitration import ArbitrationEngine, ArbitrationConfig, ArbitrationType
from traitDict import traits, personal_traits
import numpy as np

class GoalScheduler:
    def __init__(self, tick_interval: float = 1.0, progress_delta: float = 0.1, decay_rate: float = 0.05):
        self.tick_interval = tick_interval
        self.progress_delta = progress_delta
        self.decay_rate = decay_rate
        self.engine = GoalEngine(decay_rate=decay_rate)
        self.cognitive_state = CognitiveState()
        self.goal_space = GoalSpace(max_iterations=100, convergence_threshold=1e-6)
        self.ticks = 0
        self.max_goals = 10  # Limit to avoid runaway growth
        self.completed_goals = 0  # Track number of completed goals
        
        # Initialize traits from traitDict
        self.engine.traits = self._initialize_traits()

    def _initialize_traits(self) -> Dict[str, Dict[str, Any]]:
        """Initialize traits from traitDict with proper structure"""
        initialized_traits = {}
        for name, data in traits.items():
            initialized_traits[name] = {
                'value': data.get('value', 0.5),
                'description': data.get('description', ''),
                'goal_influence': data.get('goal_influence', {}),
                'links': data.get('links', []),
                'valence': data.get('valence', 0.0),
                'stability': data.get('stability', 1.0)
            }
        return initialized_traits

    def _generate_goals_from_traits(self):
        """Generate meaningful goals based on trait values and descriptions"""
        for trait, data in self.engine.traits.items():
            if data['value'] > 0.8 and f'{trait}_goal' not in self.goal_space.goals:
                # Create a goal description based on trait
                description = f"Develop and strengthen {trait} through focused practice and reflection"
                
                # Create utility function that considers trait value and linked traits
                def utility(dep_utils, weights, trait=trait, data=data):
                    base_util = 0.5 + 0.5 * data['value']
                    # Add influence from linked traits
                    for linked_trait in data.get('links', []):
                        if linked_trait in self.engine.traits:
                            base_util += 0.2 * self.engine.traits[linked_trait]['value']
                    return min(1.0, base_util)
                
                # Create goal with rich attributes
                goal = Goal(
                    name=f'{trait}_goal',
                    attributes={
                        trait: data['value'],
                        'description': description,
                        'linked_traits': data.get('links', [])
                    },
                    priority=data['value'],
                    dependencies=set(),
                    utility_fn=utility
                )
                try:
                    self.goal_space.add_goal(goal)
                    print(f"Generated new goal: {trait}_goal")
                    print(f"Description: {description}")
                except ValueError as e:
                    print(f"Could not add goal {trait}_goal: {e}")

    def _resolve_satisfied_goals(self):
        """Remove satisfied goals and update traits"""
        to_remove = [name for name, goal in self.goal_space.goals.items() 
                    if goal.progress >= 1.0 or goal.status == GoalStatus.SATISFIED]
        if to_remove:
            self.completed_goals += len(to_remove)  # Increment completed goals counter
            for name in to_remove:
                goal = self.goal_space.goals[name]
                trait = name.replace('_goal', '')
                print(f"\nGoal satisfied: {name}")
                print(f"Description: {goal.attributes.get('description', '')}")
                print(f"Final progress: {goal.progress:.2f}")
                
                # Remove goal
                del self.goal_space.goals[name]
                if name in self.goal_space.dependency_graph:
                    del self.goal_space.dependency_graph[name]
                if name in self.goal_space.arbitration_weights:
                    del self.goal_space.arbitration_weights[name]
            
            # Clear arbitration history when goals are removed
            self.goal_space.arbitration_engine.history = []

    def _progress_goals(self):
        """Progress the top goal and print detailed status"""
        top_goals = self.goal_space.get_top_goals(1)
        if top_goals:
            goal = top_goals[0]
            self.goal_space.update_goal_progress(goal.name, self.progress_delta)
            print(f"\nProgressing goal: {goal.name}")
            print(f"Description: {goal.attributes.get('description', '')}")
            print(f"Current progress: {goal.progress:.2f}")
            print(f"Utility: {self.goal_space.get_utility(goal.name):.3f}")
            print(f"Weight: {self.goal_space.arbitration_weights[goal.name]:.3f}")

    def _feedback_to_traits(self):
        """Update traits based on goal progress"""
        for name, goal in self.goal_space.goals.items():
            if goal.progress >= 1.0 or goal.status == GoalStatus.SATISFIED:
                trait = name.replace('_goal', '')
                if trait in self.engine.traits:
                    old_value = self.engine.traits[trait]['value']
                    self.engine.traits[trait]['value'] = min(1.0, old_value + 0.1)
                    print(f"\nTrait updated: {trait}")
                    print(f"Old value: {old_value:.2f}")
                    print(f"New value: {self.engine.traits[trait]['value']:.2f}")

    def run(self):
        print("Starting perpetual Goal Scheduler...")
        print("\nInitial Traits:")
        for name, data in self.engine.traits.items():
            print(f"{name}: {data['value']:.2f}")
            print(f"Description: {data['description']}")
            if data['links']:
                print(f"Linked traits: {', '.join(data['links'])}")
            print()
        
        try:
            while True:
                self.ticks += 1
                print(f"\n=== Tick {self.ticks} ===")
                
                # Update engine and cognitive state
                self.engine.update_goals()
                self._generate_goals_from_traits()
                self._progress_goals()
                self._resolve_satisfied_goals()
                self._feedback_to_traits()
                
                # Print system status
                print(f"\nActive goals: {len(self.goal_space.goals)}")
                for name, goal in self.goal_space.goals.items():
                    print(f"\n{name}:")
                    print(f"Description: {goal.attributes.get('description', '')}")
                    print(f"Progress: {goal.progress:.2f}")
                    print(f"Utility: {self.goal_space.get_utility(name):.3f}")
                    print(f"Weight: {self.goal_space.arbitration_weights[name]:.3f}")
                
                # Sleep for tick interval
                time.sleep(self.tick_interval)
                
        except KeyboardInterrupt:
            print("\nScheduler stopped by user.")
            print(f"Final tick count: {self.ticks}")

if __name__ == "__main__":
    # Create and run the scheduler
    scheduler = GoalScheduler(
        tick_interval=1.0,    # 1 second between ticks
        progress_delta=0.1,   # Progress goals by 10% each tick
        decay_rate=0.05       # 5% trait decay per tick
    )
    scheduler.run() 