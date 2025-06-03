from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
from dynamicUpdates import DynamicSystem, DynamicConfig, UpdateType
import math

class ArbitrationType(Enum):
    WEIGHTED = "weighted"
    SOFTMAX = "softmax"
    NASH = "nash"

@dataclass
class ArbitrationConfig:
    """Configuration for arbitration mechanisms"""
    type: ArbitrationType
    temperature: float = 1.0  # For softmax
    weights: Optional[Dict[str, float]] = None  # For weighted arbitration
    max_iterations: int = 100  # For Nash equilibrium computation
    convergence_threshold: float = 1e-6

class ArbitrationEngine:
    """
    Implementation of formal arbitration models from Section 4.
    Handles weighted utility arbitration, softmax arbitration, and Nash equilibrium computation.
    """
    def __init__(self, 
                 arbitration_rule: str = "softmax",
                 temperature: float = 1.0,
                 update_type: UpdateType = UpdateType.DISCRETE):
        """
        Initialize the arbitration engine with specified rule and parameters.
        
        Args:
            arbitration_rule: Type of arbitration rule to use ("softmax" or "nash")
            temperature: Temperature parameter for softmax (higher = more uniform)
            update_type: Whether to use discrete or continuous-time updates
        """
        self.arbitration_rule = arbitration_rule
        self.base_temperature = temperature
        self.update_type = update_type
        self.current_weights = {}
        self.dynamic_system = DynamicSystem(update_type)
        self.config = DynamicConfig(update_type=update_type)
    
    def get_effective_temperature(self, utilities: Dict[str, float]) -> float:
        """Calculate effective temperature based on utility variance."""
        if not utilities:
            return self.base_temperature
            
        # Higher variance = higher temperature (more exploration)
        utility_values = list(utilities.values())
        variance = np.var(utility_values) if len(utility_values) > 1 else 0
        return self.base_temperature * (1.0 + variance)
    
    def softmax_arbitration(self, utilities: Dict[str, float]) -> Dict[str, float]:
        """Softmax arbitration with temperature modulation."""
        if not utilities:
            return {}
            
        # Get effective temperature
        temp = self.get_effective_temperature(utilities)
        
        # Apply softmax with temperature
        max_utility = max(utilities.values())
        exps = {k: math.exp((v - max_utility) / temp) for k, v in utilities.items()}
        total = sum(exps.values())
        
        if total == 0:
            # Fallback to uniform distribution
            n = len(utilities)
            return {k: 1/n for k in utilities}
            
        return {k: v/total for k, v in exps.items()}
    
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
        if self.arbitration_rule == "nash":
            return self._nash_arbitration
        return self.softmax_arbitration
    
    def update_weights(self, utilities: Dict[str, float], current_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Update arbitration weights using stochastic dynamics."""
        if not utilities:
            return {}
            
        # Initialize or update current weights
        if current_weights is None:
            current_weights = self.current_weights.copy()
            
        # Ensure all goals have an entry in utilities
        for goal_name in current_weights:
            if goal_name not in utilities:
                utilities[goal_name] = 0.0  # Default utility for missing goals
                
        # Get effective temperature for this update
        temp = self.get_effective_temperature(utilities)
        
        # Define arbitration function with temperature
        def arbitration_fn(weights):
            return self.softmax_arbitration(utilities)
            
        # Update weights using dynamic system
        new_weights = self.dynamic_system.update(
            current_weights=current_weights,
            utilities=utilities,
            arbitration_fn=arbitration_fn
        )
        
        # Store updated weights
        self.current_weights = new_weights
        return new_weights
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics including temperature effects."""
        metrics = self.dynamic_system.get_stability_metrics()
        if self.current_weights:
            metrics['temperature'] = self.get_effective_temperature(self.current_weights)
        return metrics
    
    def get_lipschitz_bound(self) -> float:
        """Get Lipschitz bound considering temperature effects."""
        return self.dynamic_system.get_lipschitz_bound()
    
    def detect_conflicts(self) -> List[tuple]:
        """Detect potential conflicts between goals."""
        if not self.current_weights:
            return []
            
        conflicts = []
        goals = list(self.current_weights.keys())
        
        for i, g1 in enumerate(goals):
            for g2 in goals[i+1:]:
                # Check for negative correlation in weights
                if self.current_weights[g1] * self.current_weights[g2] < 0:
                    conflicts.append((g1, g2))
                    
        return conflicts 