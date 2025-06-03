from typing import Dict, List, Set, Optional, Tuple, Callable
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
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0
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
    
    def evaluate_all_utilities(self) -> Dict[str, float]:
        """
        Evaluate utilities for all goals using fixed-point iteration.
        """
        utilities = {}
        max_iterations = 100
        convergence_threshold = 1e-6
        
        # Initialize utilities
        for name in self.goals:
            utilities[name] = self.get_utility(name)
        
        # Fixed-point iteration
        for _ in range(max_iterations):
            old_utilities = utilities.copy()
            
            # Update utilities
            for name in self.goals:
                utilities[name] = self.get_utility(name)
            
            # Check convergence
            max_diff = max(abs(utilities[name] - old_utilities[name])
                          for name in utilities)
            if max_diff < convergence_threshold:
                break
        
        return utilities
    
    def arbitrate_goals(self) -> Dict[str, float]:
        """
        Update arbitration weights based on current utilities.
        """
        utilities = self.evaluate_all_utilities()
        return self.arbitration_engine.update_weights(utilities)
    
    def update_goal_progress(self, goal_name: str, delta: float):
        """
        Update progress for a goal and check satisfaction.
        """
        if goal_name in self.goals:
            goal = self.goals[goal_name]
            goal.progress = max(0.0, min(1.0, goal.progress + delta))
            
            if goal.progress >= 1.0:
                goal.status = GoalStatus.SATISFIED
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics from the arbitration engine."""
        return self.arbitration_engine.get_stability_metrics()
    
    def get_lipschitz_bound(self) -> float:
        """Get Lipschitz bound from the arbitration engine."""
        return self.arbitration_engine.get_lipschitz_bound()
    
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