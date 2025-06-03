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
        self.config = DynamicConfig(update_type=update_type, temperature=temperature)
    
    def get_effective_temperature(self, utilities: Dict[str, float]) -> float:
        """Calculate effective temperature based on utility variance."""
        if not utilities:
            return self.base_temperature
            
        # Higher variance = higher temperature (more exploration)
        utility_values = list(utilities.values())
        variance = np.var(utility_values) if len(utility_values) > 1 else 0
        return self.base_temperature * (1.0 + variance)
    
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
        if self.arbitration_rule == "nash":
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