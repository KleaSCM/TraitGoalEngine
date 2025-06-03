from typing import Dict, List, Set, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import numpy as np
from enum import Enum
from arbitration import ArbitrationEngine, ArbitrationConfig, ArbitrationType, NashConfig
from dynamicUpdates import UpdateType, DynamicSystem, DynamicConfig
from stochasticDynamics import StochasticDynamics, SDEConfig

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
    linked_traits: Dict[str, float] = None
    completed: bool = False  # Track goal completion status
    completion_time: Optional[int] = None  # Track when goal was completed
    
    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.linked_traits is None:
            self.linked_traits = {}

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

@dataclass
class GoalSpaceConfig:
    """Configuration for goal space dynamics"""
    temperature: float = 1.0
    stability_threshold: float = 0.1
    progress_threshold: float = 0.8
    sde_config: Optional[SDEConfig] = None

class GoalSpace:
    """
    Implements the goal space with dynamic arbitration and recursive utility evaluation.
    """
    def __init__(self, config: Optional[GoalSpaceConfig] = None):
        self.config = config or GoalSpaceConfig()
        self.arbitration_engine = ArbitrationEngine(
            config=ArbitrationConfig(
                temperature=self.config.temperature,
                sde_config=self.config.sde_config
            )
        )
        self.stochastic_dynamics = StochasticDynamics(config=self.config.sde_config)
        self.goals: Dict[str, Any] = {}
        self.dependencies: Dict[str, Set[str]] = {}  # Initialize dependencies dictionary
        self.utility_history: List[Dict[str, float]] = []
        self.weight_history: List[Dict[str, float]] = []
        self.current_utilities: Dict[str, float] = {}
        self.current_weights: Dict[str, float] = {}
    
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
        if not self.current_weights:
            self.current_weights = {}
        self.current_weights[goal.name] = 1.0 / (len(self.goals) + 1)  # Equal initial weight
        
        # Normalize all weights
        total = sum(self.current_weights.values())
        if total > 0:
            self.current_weights = {
                name: weight / total 
                for name, weight in self.current_weights.items()
            }
            
        # Also initialize arbitration engine weights
        if not self.arbitration_engine.current_weights:
            self.arbitration_engine.current_weights = {}
        self.arbitration_engine.current_weights[goal.name] = self.current_weights[goal.name]
    
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
        """Evaluate utilities for all goals using SDE-based evolution."""
        target_utilities = {}
        
        # Calculate target utilities based on trait values
        for goal_name, goal in self.goals.items():
            # Base utility from goal's utility function
            base_utility = goal.utility_fn({})
            
            # Add influence from linked traits
            trait_influence = 0.0
            for trait, weight in goal.linked_traits.items():
                if trait in trait_values:
                    trait_influence += weight * trait_values[trait]
            
            # Combine base utility with trait influence
            target_utilities[goal_name] = max(0.0, min(1.0, base_utility + trait_influence))
        
        # Update current utilities using SDE
        for goal_name, target_utility in target_utilities.items():
            current_utility = self.current_utilities.get(goal_name, 0.5)
            stability = goal.attributes.get('stability', 0.5)
            
            # Get goal-specific trait values
            goal_traits = {
                'resilience': goal.attributes.get('resilience', 0.5),
                'decisiveness': goal.attributes.get('decisiveness', 0.5)
            }
            
            self.current_utilities[goal_name] = self.stochastic_dynamics.utility_sde(
                current_utility=current_utility,
                target_utility=target_utility,
                stability=stability,
                temperature=self.config.temperature,
                trait_values=goal_traits
            )
        
        return self.current_utilities
    
    def arbitrate_goals(self, trait_values: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Arbitrate between goals using SDE-based weight evolution."""
        if trait_values is None:
            trait_values = {}
        
        # Get current utilities
        utilities = self.evaluate_all_utilities(trait_values)
        
        # Calculate target weights using softmax
        target_weights = self.arbitration_engine.arbitrate(utilities)
        
        # Get system-wide trait values
        system_traits = {
            'resilience': np.mean([g.attributes.get('resilience', 0.5) for g in self.goals.values()]),
            'decisiveness': np.mean([g.attributes.get('decisiveness', 0.5) for g in self.goals.values()])
        }
        
        # Update current weights using SDE
        self.current_weights = self.stochastic_dynamics.weight_sde(
            current_weights=self.current_weights,
            target_weights=target_weights,
            utilities=utilities,
            temperature=self.config.temperature,
            trait_values=system_traits
        )
        
        return self.current_weights
    
    def update_goal_progress(self, goal_name: str, progress_delta: float, trait_values: Optional[Dict[str, float]] = None) -> None:
        """Update goal progress and trait values."""
        if goal_name not in self.goals:
            return
        
        goal = self.goals[goal_name]
        old_progress = goal.progress
        
        # Update progress
        goal.progress = min(1.0, max(0.0, goal.progress + progress_delta))
        
        # Check for completion
        if goal.progress >= self.config.progress_threshold and not goal.completed:
            goal.completed = True
            goal.completion_time = len(self.utility_history)
        
        # Update stability metrics
        if trait_values is None:
            trait_values = {trait: 0.5 for trait in goal.linked_traits} if goal.linked_traits else {}
        self.utility_history.append(self.evaluate_all_utilities(trait_values))
        self.weight_history.append(self.arbitrate_goals(trait_values))
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics for the goal space."""
        if not self.utility_history or not self.weight_history:
            return {}
        
        # Calculate utility stability
        recent_utilities = self.utility_history[-10:]
        if recent_utilities:
            # Ensure all utility dictionaries have the same keys
            all_keys = set().union(*[u.keys() for u in recent_utilities])
            utility_arrays = []
            for u in recent_utilities:
                # Fill missing values with 0.5 (neutral utility)
                values = [u.get(k, 0.5) for k in all_keys]
                utility_arrays.append(values)
            utility_variance = np.var(utility_arrays) if utility_arrays else 0.0
        else:
            utility_variance = 0.0
        
        # Calculate weight stability
        recent_weights = self.weight_history[-10:]
        if recent_weights:
            # Ensure all weight dictionaries have the same keys
            all_keys = set().union(*[w.keys() for w in recent_weights])
            weight_arrays = []
            for w in recent_weights:
                # Fill missing values with 1/n (equal weights)
                n = len(all_keys)
                values = [w.get(k, 1.0/n) for k in all_keys]
                weight_arrays.append(values)
            weight_variance = np.var(weight_arrays) if weight_arrays else 0.0
        else:
            weight_variance = 0.0
        
        # Get SDE stability metrics
        sde_metrics = self.stochastic_dynamics.get_stability_metrics()
        
        return {
            'utility_stability': 1.0 / (1.0 + utility_variance),
            'weight_stability': 1.0 / (1.0 + weight_variance),
            **sde_metrics
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