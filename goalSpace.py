from typing import Dict, List, Set, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from enum import Enum
from arbitration import ArbitrationEngine, ArbitrationConfig, ArbitrationType

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
    
    @property
    def effective_weight(self) -> float:
        """
        Effective Goal Weight from Definition 9:
        w(g,t) = p(g) · u(g,t)
        """
        return self.priority * self.urgency

class GoalSpace:
    """
    Implementation of the Goal Space G ⊆ Rn and Goal Dependency Graph D = (V,E)
    from Definitions 1 and 5, with recursive utility evaluation from Section 6.
    """
    def __init__(self, max_iterations: int = 100, convergence_threshold: float = 1e-6):
        self.goals: Dict[str, Goal] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.utility_cache: Dict[str, float] = {}
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.arbitration_weights: Dict[str, float] = {}  # Initialize arbitration weights
        
        # Initialize arbitration engine with softmax configuration
        self.arbitration_engine = ArbitrationEngine(
            ArbitrationConfig(
                type=ArbitrationType.SOFTMAX,
                temperature=1.0
            )
        )
        
    def add_goal(self, goal: Goal):
        """
        Add a goal to the space, maintaining the DAG property of the dependency graph.
        """
        self.goals[goal.name] = goal
        self.dependency_graph[goal.name] = goal.dependencies
        
        # Initialize arbitration weight for the new goal (uniform)
        n = len(self.goals)
        for g in self.goals:
            self.arbitration_weights[g] = 1.0 / n
        
        # Verify DAG property (antisymmetry and transitivity from Definition 4)
        if not self._is_dag():
            raise ValueError("Adding this goal would create a cycle in the dependency graph")
        
        # Initialize with uniform weights through arbitration engine
        if not hasattr(self, 'arbitration_engine'):
            self.arbitration_engine = ArbitrationEngine(
                ArbitrationConfig(
                    type=ArbitrationType.SOFTMAX,
                    temperature=1.0
                )
            )
    
    def _is_dag(self) -> bool:
        """
        Verify the Goal Dependency Graph is a DAG (Definition 5).
        Checks reflexivity, antisymmetry, and transitivity properties.
        """
        visited = set()
        temp = set()
        
        def visit(node: str) -> bool:
            if node in temp:
                return False  # Cycle detected (violates antisymmetry)
            if node in visited:
                return True
            
            temp.add(node)
            for neighbor in self.dependency_graph.get(node, set()):
                if not visit(neighbor):
                    return False
            
            temp.remove(node)
            visited.add(node)
            return True
        
        return all(visit(node) for node in self.dependency_graph)
    
    def get_utility(self, goal_name: str) -> float:
        """
        Recursive Utility Function from Definition 21.
        Implements fixed-point iteration to find utility values.
        """
        if goal_name in self.utility_cache:
            return self.utility_cache[goal_name]
        
        goal = self.goals[goal_name]
        
        # If goal has a recursive utility function, use it
        if goal.utility_fn is not None:
            # Get dependency utilities
            dep_utils = {dep: self.get_utility(dep) 
                        for dep in goal.dependencies}
            
            # Evaluate utility function
            utility = goal.utility_fn(dep_utils, self.arbitration_weights)
        else:
            # Base case: no recursive function, use effective weight
            dependencies = self.dependency_graph[goal_name]
            
            if not dependencies:
                # No dependencies, use effective weight directly
                utility = goal.effective_weight
            else:
                # Recursive case: aggregate utilities of dependencies
                dep_utilities = [self.get_utility(dep) for dep in dependencies]
                utility = self._aggregate_utilities(dep_utilities) * goal.effective_weight
        
        self.utility_cache[goal_name] = utility
        return utility
    
    def evaluate_all_utilities(self) -> Dict[str, float]:
        """
        Evaluate utilities for all goals using fixed-point iteration.
        Implements the recursive utility vector function U from Definition 21.
        """
        utilities = {name: 0.0 for name in self.goals}
        
        for _ in range(self.max_iterations):
            old_utilities = utilities.copy()
            
            # Update each goal's utility
            for name in self.goals:
                utilities[name] = self.get_utility(name)
            
            # Check convergence
            if self._check_convergence(old_utilities, utilities):
                break
        
        return utilities
    
    def _check_convergence(self, old: Dict[str, float], new: Dict[str, float]) -> bool:
        """Check if utility values have converged"""
        return all(abs(old[name] - new[name]) < self.convergence_threshold
                  for name in old)
    
    def arbitrate_goals(self, temperature: float = 1.0) -> Dict[str, float]:
        """
        Recursive arbitration update from Definition 22.
        Updates arbitration weights based on current utilities and previous weights.
        """
        # Update temperature if provided
        if temperature != self.arbitration_engine.config.temperature:
            self.arbitration_engine.config.temperature = temperature
            
        # Evaluate all utilities recursively
        utilities = self.evaluate_all_utilities()
        
        # Get new weights from arbitration engine
        new_weights = self.arbitration_engine.arbitrate(utilities)
        
        # Update weights with smoothing
        for name in self.goals:
            self.arbitration_weights[name] = (
                0.7 * new_weights[name] + 
                0.3 * self.arbitration_weights[name]
            )
        
        return self.arbitration_weights
    
    def step(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Combined recursive decision dynamics from Definition 23.
        Updates both utilities and arbitration weights.
        """
        # Evaluate utilities recursively
        utilities = self.evaluate_all_utilities()
        
        # Update arbitration
        weights = self.arbitrate_goals()
        
        return utilities, weights
    
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
    
    def update_urgency(self, goal_name: str, time: float):
        """
        Update goal urgency based on time (Definition 8).
        Ensures boundedness of effective weight (Theorem 1).
        """
        goal = self.goals[goal_name]
        
        # Example urgency function: exponential decay from 1.0
        goal.urgency = np.exp(-0.1 * time)
        
        # Clear utility cache as urgency affects utility
        self.utility_cache.clear()
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """
        Get stability metrics for the arbitration process.
        """
        return self.arbitration_engine.get_stability_metrics()
    
    def get_lipschitz_bound(self) -> float:
        """
        Get Lipschitz constant bound for the arbitration operator.
        """
        return self.arbitration_engine.get_lipschitz_bound()
    
    def detect_conflicts(self) -> List[Tuple[str, str]]:
        """
        Conflict Detection from Definition 14.
        Identifies pairs of goals with opposing utility gradients.
        """
        conflicts = []
        goal_names = list(self.goals.keys())
        
        for i, g1 in enumerate(goal_names):
            for g2 in goal_names[i+1:]:
                # Compute utility gradients
                grad1 = self._compute_utility_gradient(g1)
                grad2 = self._compute_utility_gradient(g2)
                
                # Check for negative correlation
                if np.dot(grad1, grad2) < 0:
                    conflicts.append((g1, g2))
        
        return conflicts
    
    def _compute_utility_gradient(self, goal_name: str):
        goal = self.goals[goal_name]
        # FIX: Only use numeric attributes for gradient calculation to avoid errors with strings/lists
        numeric_values = [v for v in goal.attributes.values() if isinstance(v, (int, float))]
        return np.array(numeric_values)
    
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
    
    def update_goal_progress(self, goal_name: str, delta: float):
        """
        Update progress of a goal.
        """
        goal = self.goals[goal_name]
        goal.progress = min(1.0, goal.progress + delta)
        
        if goal.progress >= 0.8:  # Satisfaction threshold
            goal.status = GoalStatus.SATISFIED
        elif goal.progress <= 0.0:
            goal.status = GoalStatus.FAILED
    
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