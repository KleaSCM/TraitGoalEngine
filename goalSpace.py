from typing import Dict, List, Set, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import numpy as np
from enum import Enum
from arbitration import ArbitrationEngine, ArbitrationConfig, ArbitrationType
from dynamicUpdates import UpdateType

class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SATISFIED = "satisfied"
    FAILED = "failed"

@dataclass
class Goal:
    """
    Goal representation following Definition 1 (Goal Space) and Definition 2 (Symbolic Goal Encoding).
    Each goal is represented as a vector in Rn and can be composed with other goals.
    """
    name: str
    attributes: Dict[str, float]  # Vector representation g = (g1, g2, ..., gn)
    priority: float  # Static priority p(g) from Definition 7
    dependencies: Set[str]  # Goal Dependency Partial Order ⪯ from Definition 4
    utility_fn: Optional[Callable[[Dict[str, float], Dict[str, float]], float]] = None  # Recursive utility function
    status: GoalStatus = GoalStatus.ACTIVE
    progress: float = 0.0
    satisfaction_threshold: float = 0.8
    utility: float = 0.5  # Initial utility
    reversion_rate: float = 0.1  # Mean reversion rate
    noise_scale: float = 0.05  # Noise scale for utility process
    urgency: float = 0.0  # Time-dependent urgency u(g,t) from Definition 8
    description: str = ""
    linked_traits: List[str] = None
    
    @property
    def effective_weight(self) -> float:
        """
        Effective Goal Weight from Definition 9:
        w(g,t) = p(g) · u(g,t)
        """
        return self.priority * self.urgency

    def update_utility(self, trait_values: Dict[str, float]) -> float:
        """Update utility using mean-reverting process."""
        # Calculate target utility based on trait values
        target_utility = self.priority * (0.5 + 0.5 * self.progress)
        
        # Mean-reverting update with noise
        noise = np.random.normal(0, self.noise_scale)
        self.utility = self.utility + self.reversion_rate * (target_utility - self.utility) + noise
        
        # Clamp utility to [0, 1]
        self.utility = max(0.0, min(1.0, self.utility))
        return self.utility

class GoalSpace:
    """
    Implements the goal space with dynamic arbitration and recursive utility evaluation.
    """
    def __init__(self, 
                 update_type: UpdateType = UpdateType.DISCRETE,
                 arbitration_rule: str = "softmax",
                 temperature: float = 1.0):
        self.goals: Dict[str, Goal] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.arbitration_engine = ArbitrationEngine(
            arbitration_rule=arbitration_rule,
            temperature=temperature,
            update_type=update_type
        )
        self.utility_history: List[Dict[str, float]] = []
        self.weight_history: List[Dict[str, float]] = []
        self.temperature = temperature  # Base temperature for softmax
    
    def add_goal(self, goal: Goal, dependencies: Optional[List[str]] = None):
        """
        Add a goal to the space with optional dependencies.
        Checks for cycles in the dependency graph.
        """
        # Check for cycles
        if dependencies:
            self.dependencies[goal.name] = set(dependencies)
            if self._has_cycle(goal.name):
                raise ValueError(f"Adding {goal.name} would create a cycle in the dependency graph")
        else:
            self.dependencies[goal.name] = set()
        
        self.goals[goal.name] = goal
        
        # Initialize weights for the new goal
        if not self.arbitration_engine.current_weights:
            self.arbitration_engine.current_weights = {}
        self.arbitration_engine.current_weights[goal.name] = 1.0 / (len(self.goals) + 1)  # Equal initial weight
        
        # Normalize all weights
        total = sum(self.arbitration_engine.current_weights.values())
        if total > 0:
            self.arbitration_engine.current_weights = {
                name: weight / total 
                for name, weight in self.arbitration_engine.current_weights.items()
            }
    
    def _has_cycle(self, start: str) -> bool:
        """Check if adding a goal would create a cycle in the dependency graph."""
        visited = set()
        path = set()
        
        def dfs(node: str) -> bool:
            if node in path:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            path.add(node)
            
            for dep in self.dependencies.get(node, set()):
                if dep in self.goals and dfs(dep):
                    return True
            
            path.remove(node)
            return False
        
        return dfs(start)
    
    def get_utility(self, goal_name: str) -> float:
        """
        Recursively compute utility for a goal based on its dependencies.
        """
        if goal_name not in self.goals:
            return 0.0
        
        goal = self.goals[goal_name]
        
        # If goal has a utility function, use it
        if goal.utility_fn:
            # Get utilities of dependencies
            dep_utilities = {
                dep: self.get_utility(dep)
                for dep in self.dependencies[goal_name]
                if dep in self.goals
            }
            return goal.utility_fn(dep_utilities)
        
        # Default utility based on progress and priority
        return goal.priority * (0.5 + 0.5 * goal.progress)  # Add base utility of 0.5
    
    def evaluate_all_utilities(self, trait_values: Dict[str, float]) -> Dict[str, float]:
        """Evaluate utilities for all goals using fixed-point iteration."""
        if not self.goals:
            return {}
            
        # Update utilities for all goals
        utilities = {}
        for goal in self.goals.values():
            utilities[goal.name] = goal.update_utility(trait_values)
            
        # Record utility history
        self.utility_history.append(utilities)
        return utilities
    
    def arbitrate_goals(self, trait_values: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Update arbitration weights based on current utilities.
        
        Args:
            trait_values: Optional dictionary of trait values. If not provided,
                         will use default values of 0.5 for all traits.
        """
        if trait_values is None:
            # Get trait values from goals' linked traits
            trait_values = {}
            for goal in self.goals.values():
                if goal.linked_traits:
                    for trait in goal.linked_traits:
                        if trait not in trait_values:
                            trait_values[trait] = 0.5  # Default trait value
        
        utilities = self.evaluate_all_utilities(trait_values)
        return self.arbitration_engine.update_weights(utilities)
    
    def update_goal_progress(self, goal_name: str, progress_delta: float) -> None:
        """Update progress for a goal."""
        if goal_name in self.goals:
            goal = self.goals[goal_name]
            goal.progress = max(0.0, min(1.0, goal.progress + progress_delta))
            
            # Update status if threshold reached
            if goal.progress >= goal.satisfaction_threshold:
                goal.status = GoalStatus.SATISFIED
    
    def get_effective_temperature(self) -> float:
        """Calculate effective temperature based on utility variance."""
        if not self.utility_history:
            return self.temperature
            
        # Calculate variance of utilities
        recent_utilities = self.utility_history[-10:]  # Look at last 10 steps
        if not recent_utilities:
            return self.temperature
            
        # Compute variance across all goals
        all_utilities = []
        for util_dict in recent_utilities:
            all_utilities.extend(util_dict.values())
            
        variance = np.var(all_utilities) if all_utilities else 0.0
        
        # Temperature increases with variance
        return self.temperature * (1.0 + variance)
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics for the goal space."""
        if len(self.utility_history) < 2:
            return {
                'max_diff': 0.0,
                'mean_diff': 0.0,
                'std_diff': 0.0,
                'convergence_rate': 0.0
            }
            
        # Get last two utility vectors
        old = self.utility_history[-2]
        new = self.utility_history[-1]
        
        # Compute differences
        diffs = [abs(old[name] - new[name]) for name in old]
        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)
        std_diff = np.std(diffs)
        
        # Compute convergence rate
        if len(self.utility_history) > 2:
            prev_diff = sum(abs(self.utility_history[-3][name] - old[name]) 
                          for name in old)
            curr_diff = sum(diffs)
            convergence_rate = (prev_diff - curr_diff) / prev_diff if prev_diff > 0 else 0.0
        else:
            convergence_rate = 0.0
            
        return {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'convergence_rate': convergence_rate
        }
    
    def get_lipschitz_bound(self) -> float:
        """Get Lipschitz bound for the goal space."""
        if len(self.utility_history) < 2:
            return float('inf')
            
        diffs = []
        for i in range(1, len(self.utility_history)):
            old = self.utility_history[i-1]
            new = self.utility_history[i]
            diff = sum(abs(old[name] - new[name]) for name in old)
            diffs.append(diff)
            
        return max(diffs) if diffs else float('inf')
    
    def detect_conflicts(self) -> List[tuple]:
        """Detect potential conflicts between goals."""
        return self.arbitration_engine.detect_conflicts()
    
    def get_top_goals(self, n: int = 5) -> List[Goal]:
        """
        Get top n goals by utility after arbitration.
        """
        # Handle empty goal set
        if not self.goals:
            return []
            
        # Get arbitrated weights
        weights = self.arbitrate_goals()
        
        # Sort by weighted utility
        sorted_goals = sorted(
            self.goals.values(),
            key=lambda g: weights[g.name] * self.get_utility(g.name),
            reverse=True
        )
        return sorted_goals[:n]
    
    def get_goal_vector(self, goal_name: str) -> np.ndarray:
        """
        Convert goal to vector representation (Definition 1).
        """
        goal = self.goals[goal_name]
        vectors = []
        
        # Add attribute vector
        vectors.extend(goal.attributes.values())
        
        # Add priority and urgency
        vectors.extend([goal.priority, goal.urgency])
        
        # Add progress
        vectors.append(goal.progress)
        
        return np.array(vectors)
    
    def get_goal_space_vector(self) -> np.ndarray:
        """
        Convert entire goal space to vector representation.
        """
        vectors = []
        for goal in self.goals.values():
            vectors.extend(self.get_goal_vector(goal.name))
        return np.array(vectors)
    
    def update_urgency(self, goal_name: str, time: float):
        """
        Update goal urgency based on time (Definition 8).
        Ensures boundedness of effective weight (Theorem 1).
        """
        goal = self.goals[goal_name]
        
        # Example urgency function: exponential decay from 1.0
        goal.urgency = np.exp(-0.1 * time)
    
    def _aggregate_utilities(self, utilities: List[float]) -> float:
        """
        Utility Aggregation Operator from Definition 6.
        Implements monotonicity property.
        """
        # Using weighted geometric mean as aggregation operator
        # This ensures monotonicity and handles zero utilities gracefully
        if not utilities:
            return 0.0
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        utilities = [u + eps for u in utilities]
        
        # Compute geometric mean
        return np.exp(np.mean(np.log(utilities))) - eps
    
    def _compute_utility_gradient(self, goal_name: str):
        goal = self.goals[goal_name]
        # FIX: Only use numeric attributes for gradient calculation to avoid errors with strings/lists
        numeric_values = [v for v in goal.attributes.values() if isinstance(v, (int, float))]
        return np.array(numeric_values) 