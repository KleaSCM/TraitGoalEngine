from typing import Dict, List, Tuple, Optional, Callable, Set
import numpy as np
from dataclasses import dataclass
from enum import Enum
from dynamicUpdates import DynamicSystem, DynamicConfig, UpdateType
import math
from stochasticDynamics import StochasticDynamics, SDEConfig

class ArbitrationType(Enum):
    WEIGHTED = "weighted"
    SOFTMAX = "softmax"
    NASH = "nash"

@dataclass
class ArbitrationConfig:
    """Configuration for arbitration engine"""
    temperature: float = 1.0
    update_type: UpdateType = UpdateType.DISCRETE
    sde_config: Optional[SDEConfig] = None

@dataclass
class NashConfig:
    """Configuration for Nash equilibrium computation"""
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    learning_rate: float = 0.1
    min_utility: float = 1e-6  # Minimum utility to avoid zero division

class ArbitrationEngine:
    """
    Implements continuous-time arbitration between goals using stochastic differential equations.
    Includes Nash equilibrium computation for conflict resolution.
    """
    def __init__(self, config: Optional[ArbitrationConfig] = None):
        self.config = config or ArbitrationConfig()
        self.stochastic_dynamics = StochasticDynamics(config=self.config.sde_config)
        self.current_weights: Dict[str, float] = {}
        self.nash_config = NashConfig()
        self.conflict_groups: List[Set[str]] = []
    
    def arbitrate(self, utilities: Dict[str, float], trait_values: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Arbitrate between goals using SDE-based weight evolution and Nash equilibrium for conflicts.
        """
        if not utilities:
            return {}
        
        # Detect conflicts
        conflict_groups = self.detect_conflicts()
        
        # Initialize weights
        weights = {}
        
        # Handle each conflict group separately
        for group in conflict_groups:
            # Compute Nash equilibrium for conflicting goals
            group_weights = self.compute_nash_equilibrium(utilities, group)
            weights.update(group_weights)
        
        # Handle non-conflicting goals using softmax
        non_conflicting = set(utilities.keys()) - set().union(*conflict_groups)
        if non_conflicting:
            # Get effective temperature based on traits
            effective_temp = self.stochastic_dynamics.get_effective_temperature(trait_values or {})
            
            # Calculate softmax weights for non-conflicting goals
            non_conflict_weights = self._softmax(
                {g: utilities[g] for g in non_conflicting},
                effective_temp
            )
            weights.update(non_conflict_weights)
        
        # Update current weights using SDE
        self.current_weights = self.stochastic_dynamics.weight_sde(
            current_weights=self.current_weights,
            target_weights=weights,
            utilities=utilities,
            temperature=self.stochastic_dynamics.get_effective_temperature(trait_values or {}),
            trait_values=trait_values
        )
        
        return self.current_weights
    
    def _softmax(self, values: Dict[str, float], temperature: float) -> Dict[str, float]:
        """Convert utilities to probabilities using softmax with temperature."""
        if not values:
            return {}
        
        # Scale by temperature
        scaled = {k: v/temperature for k, v in values.items()}
        
        # Subtract max for numerical stability
        max_val = max(scaled.values())
        exps = {k: np.exp(v - max_val) for k, v in scaled.items()}
        
        # Normalize
        total = sum(exps.values())
        if total == 0:
            return {k: 1.0/len(values) for k in values}
        
        return {k: v/total for k, v in exps.items()}
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics for the arbitration process."""
        return self.stochastic_dynamics.get_stability_metrics()
    
    def get_effective_temperature(self, utilities: Dict[str, float]) -> float:
        """Calculate effective temperature based on utility variance."""
        if not utilities:
            return self.config.temperature
            
        # Higher variance = higher temperature (more exploration)
        utility_values = list(utilities.values())
        variance = np.var(utility_values) if len(utility_values) > 1 else 0
        return self.config.temperature * (1.0 + variance)
    
    def softmax_arbitration(self, utilities: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
        """
        Apply softmax with temperature modulation to utilities.
        Temperature is modulated by trait states (resilience and decisiveness).
        """
        if not utilities:
            return {}
        
        # Apply temperature scaling
        scaled_utilities = {k: v / temperature for k, v in utilities.items()}
        
        # Compute softmax
        max_util = max(scaled_utilities.values())
        exps = {k: math.exp(v - max_util) for k, v in scaled_utilities.items()}
        total = sum(exps.values())
        
        if total == 0:
            # Avoid division by zero; fallback to uniform distribution
            n = len(utilities)
            return {k: 1/n for k in utilities}
        
        return {k: v / total for k, v in exps.items()}
    
    def _nash_arbitration(self, utilities: Dict[str, float]) -> Dict[str, float]:
        """
        Nash arbitration rule that maximizes the product of utilities.
        This implements a form of proportional fairness.
        """
        if not utilities:
            return {}
        
        # Ensure all utilities are positive
        min_util = min(utilities.values())
        if min_util < 0:
            utilities = {k: v - min_util + 1e-6 for k, v in utilities.items()}
        
        # Compute weights proportional to utilities
        total = sum(utilities.values())
        if total == 0:
            n = len(utilities)
            return {k: 1/n for k in utilities}
        
        return {k: v/total for k, v in utilities.items()}
    
    def get_arbitration_rule(self) -> Callable:
        """Get the appropriate arbitration rule function."""
        if self.config.update_type == UpdateType.NASH:
            return self._nash_arbitration
        return self.softmax_arbitration
    
    def update_weights(self, utilities: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights based on utilities and trait states.
        Implements a continuous-time weight update process.
        """
        if not utilities:
            return {}
        
        # Initialize or update current weights
        if not self.current_weights:
            # First time: initialize with uniform distribution
            n = len(utilities)
            self.current_weights = {k: 1/n for k in utilities}
        else:
            # Ensure all goals have an entry in utilities
            for goal_name in self.current_weights:
                if goal_name not in utilities:
                    utilities[goal_name] = 0.0  # Default utility for missing goals
        
        # Calculate weight updates using continuous-time dynamics
        new_weights = {}
        total_utility = sum(utilities.values())
        
        if total_utility > 0:
            for goal_name, utility in utilities.items():
                current_weight = self.current_weights.get(goal_name, 0.0)
                target_weight = utility / total_utility
                
                # Continuous-time update with trait modulation
                weight_delta = 0.1 * (target_weight - current_weight)  # Learning rate
                new_weights[goal_name] = max(0.0, min(1.0, current_weight + weight_delta))
        else:
            # If no utility, maintain current weights
            new_weights = self.current_weights.copy()
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v/total_weight for k, v in new_weights.items()}
        
        # Update current weights
        self.current_weights = new_weights
        
        return new_weights
    
    def get_lipschitz_bound(self) -> float:
        """Get Lipschitz bound considering temperature effects."""
        return self.stochastic_dynamics.get_lipschitz_bound()
    
    def detect_conflicts(self) -> List[Set[str]]:
        """
        Detect groups of conflicting goals using utility correlation.
        Goals are in conflict if their utilities are negatively correlated.
        """
        if not self.current_weights:
            return []
        
        # Build correlation matrix
        goals = list(self.current_weights.keys())
        n = len(goals)
        correlation_matrix = np.zeros((n, n))
        
        # Calculate correlations between goals
        for i, g1 in enumerate(goals):
            for j, g2 in enumerate(goals[i+1:], i+1):
                # Get utility histories for both goals
                g1_utils = [w[g1] for w in self.weight_history if g1 in w]
                g2_utils = [w[g2] for w in self.weight_history if g2 in w]
                
                if len(g1_utils) > 1 and len(g2_utils) > 1:
                    # Calculate correlation
                    correlation = np.corrcoef(g1_utils, g2_utils)[0, 1]
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation
        
        # Find connected components with negative correlations
        visited = set()
        conflict_groups = []
        
        def dfs(node: int, group: Set[str]):
            visited.add(node)
            group.add(goals[node])
            
            for j in range(n):
                if j not in visited and correlation_matrix[node, j] < -0.3:  # Threshold for conflict
                    dfs(j, group)
        
        for i in range(n):
            if i not in visited:
                group = set()
                dfs(i, group)
                if len(group) > 1:  # Only add groups with multiple goals
                    conflict_groups.append(group)
        
        self.conflict_groups = conflict_groups
        return conflict_groups
    
    def compute_nash_equilibrium(self, utilities: Dict[str, float], conflict_group: Set[str]) -> Dict[str, float]:
        """
        Compute Nash equilibrium for a group of conflicting goals.
        Uses gradient ascent to maximize the product of utilities.
        """
        if not conflict_group:
            return {}
        
        # Initialize weights uniformly
        weights = {g: 1.0/len(conflict_group) for g in conflict_group}
        
        for _ in range(self.nash_config.max_iterations):
            # Calculate current utilities
            current_utils = {g: utilities[g] * weights[g] for g in conflict_group}
            
            # Calculate gradients
            gradients = {}
            for g in conflict_group:
                # Gradient is the product of other utilities
                other_utils = [current_utils[h] for h in conflict_group if h != g]
                gradients[g] = np.prod(other_utils) if other_utils else 1.0
            
            # Update weights using gradient ascent
            new_weights = {}
            for g in conflict_group:
                new_weights[g] = weights[g] + self.nash_config.learning_rate * gradients[g]
            
            # Project onto simplex (ensure weights sum to 1)
            total = sum(new_weights.values())
            if total > 0:
                new_weights = {g: w/total for g, w in new_weights.items()}
            
            # Check convergence
            if all(abs(new_weights[g] - weights[g]) < self.nash_config.convergence_threshold 
                  for g in conflict_group):
                break
            
            weights = new_weights
        
        return weights
    
    def get_conflict_metrics(self) -> Dict[str, float]:
        """Get metrics about goal conflicts."""
        if not self.conflict_groups:
            return {'num_conflicts': 0, 'avg_conflict_size': 0}
        
        return {
            'num_conflicts': len(self.conflict_groups),
            'avg_conflict_size': np.mean([len(g) for g in self.conflict_groups]),
            'max_conflict_size': max(len(g) for g in self.conflict_groups)
        } 