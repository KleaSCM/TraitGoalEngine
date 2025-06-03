# goalsEngine.py

import math
from typing import Dict, Any, List, Optional
from traitGoalEngine import prepare_traits_for_goals_engine, calculate_goal_priorities

def sigmoid(x: float) -> float:
    """Smooth nonlinear activation saturating between 0 and 1."""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def softmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Convert raw scores to normalized probabilities using softmax."""
    if not scores:
        return {}
    max_score = max(scores.values())
    exps = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exps.values())
    if total == 0:
        # Avoid division by zero; fallback to uniform distribution
        n = len(scores)
        return {k: 1/n for k in scores}
    return {k: v / total for k, v in exps.items()}

class Goal:
    def __init__(self, name: str, initial_priority: float = 0.0):
        self.name = name
        self.priority = initial_priority  # softmax-normalized priority
        self.progress = 0.0  # 0.0 - 1.0 scale
        self.history: List[float] = []  # track priority over time
        self.satisfaction_threshold = 0.8  # threshold for goal satisfaction

    def update_progress(self, delta: float):
        self.progress = min(1.0, max(0.0, self.progress + delta))

    def record_priority(self, priority: float):
        self.priority = priority
        self.history.append(priority)

    def is_satisfied(self) -> bool:
        return self.progress >= self.satisfaction_threshold

    def __repr__(self):
        return (f"Goal(name={self.name}, priority={self.priority:.3f}, "
                f"progress={self.progress:.2f})")

class GoalEngine:
    def __init__(self,
                 decay_rate: float = 0.0,
                 sigmoid_stretch: float = 5.0,
                 sigmoid_offset: float = 2.5,
                 influence_power: float = 1.5):
        """
        Args:
            decay_rate: Trait decay [0,1].
            sigmoid_stretch: Controls sigmoid steepness.
            sigmoid_offset: Sigmoid offset to center activation.
            influence_power: Exponent applied to trait weights before scaling influence.
        """
        self.decay_rate = decay_rate
        self.sigmoid_stretch = sigmoid_stretch
        self.sigmoid_offset = sigmoid_offset
        self.influence_power = influence_power
        self.goals: Dict[str, Goal] = {}
        
        # Initialize with traits from traitGoalEngine
        self.traits = prepare_traits_for_goals_engine()

    def update_goals(self):
        """
        Recalculate goal priorities using traitGoalEngine's processing pipeline.
        Updates traits based on goal progress and generates new goals.
        """
        priorities, updated_traits = calculate_goal_priorities(decay_rate=self.decay_rate)
        
        # Update traits with new values
        self.traits = updated_traits
        
        # Update existing goals and create new ones if needed
        for name, prio in priorities.items():
            if name not in self.goals:
                self.goals[name] = Goal(name, prio)
            else:
                self.goals[name].record_priority(prio)
                self.goals[name].priority = prio

    def select_top_goals(self, n: int = 3) -> List[Goal]:
        """
        Select top n goals by priority.
        """
        sorted_goals = sorted(self.goals.values(), key=lambda g: g.priority, reverse=True)
        return sorted_goals[:n]

    def progress_goal(self, goal_name: str, delta: float):
        """
        Increment progress on a goal.
        """
        if goal_name in self.goals:
            self.goals[goal_name].update_progress(delta)

    def print_active_goals(self, n: int = 3):
        """
        Print top n active goals with progress and satisfaction.
        """
        top = self.select_top_goals(n)
        print("Active Goals:")
        for g in top:
            status = "✓" if g.is_satisfied() else "✗"
            print(f"{status} {g.name}: Priority={g.priority:.3f}, Progress={g.progress:.2f}")

if __name__ == "__main__":
    # Create and test the goal engine
    engine = GoalEngine(decay_rate=0.05)
    
    # Update goals based on traits
    engine.update_goals()
    engine.print_active_goals()
    
    # Simulate some progress on a goal
    if engine.goals:
        first_goal = next(iter(engine.goals.values()))
        engine.progress_goal(first_goal.name, 0.15)
        print("\nAfter Progress Update:")
        engine.print_active_goals()
