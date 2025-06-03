from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
from dynamicUpdates import DynamicUpdateSystem, DynamicConfig, UpdateType

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
        self.temperature = temperature
        self.weight_history: List[Dict[str, float]] = []
        self.utility_history: List[Dict[str, float]] = []
        
        # Initialize dynamic update system
        config = DynamicConfig(
            update_type=update_type,
            time_step=0.1,
            max_iterations=100,
            convergence_threshold=1e-6
        )
        self.dynamic_system = DynamicUpdateSystem(config)
    
    def _softmax(self, utilities: Dict[str, float]) -> Dict[str, float]:
        """Convert utilities to weights using softmax normalization."""
        if not utilities:
            return {}
        
        # Scale by temperature
        scaled = {k: v/self.temperature for k, v in utilities.items()}
        
        # Compute softmax
        max_util = max(scaled.values())
        exps = {k: np.exp(v - max_util) for k, v in scaled.items()}
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
        return self._softmax
    
    def update_weights(self, utilities: Dict[str, float]) -> Dict[str, float]:
        """
        Update arbitration weights using dynamic update rules.
        
        Args:
            utilities: Current goal utilities
            
        Returns:
            Updated arbitration weights
        """
        if not utilities:
            return {}
        
        # Get current weights (or initialize if first update)
        current_weights = self.weight_history[-1] if self.weight_history else {
            k: 1.0/len(utilities) for k in utilities
        }
        
        # Get arbitration rule
        arbitration_fn = self.get_arbitration_rule()
        
        # Update weights using dynamic system
        new_weights = self.dynamic_system.update(
            current_weights=current_weights,
            utilities=utilities,
            arbitration_fn=arbitration_fn
        )
        
        # Record history
        self.weight_history.append(new_weights)
        self.utility_history.append(utilities)
        
        return new_weights
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics from the dynamic system."""
        return self.dynamic_system.get_stability_metrics()
    
    def get_lipschitz_bound(self) -> float:
        """Get Lipschitz bound from the dynamic system."""
        return self.dynamic_system.get_lipschitz_bound()
    
    def detect_conflicts(self, threshold: float = 0.3) -> List[tuple]:
        """
        Detect potential conflicts between goals based on utility patterns.
        
        Args:
            threshold: Minimum correlation threshold for conflict detection
            
        Returns:
            List of (goal1, goal2) pairs that may be in conflict
        """
        if len(self.utility_history) < 2:
            return []
        
        conflicts = []
        goals = list(self.utility_history[0].keys())
        
        # Compute utility correlations
        for i in range(len(goals)):
            for j in range(i+1, len(goals)):
                g1, g2 = goals[i], goals[j]
                
                # Get utility histories
                u1 = [h[g1] for h in self.utility_history]
                u2 = [h[g2] for h in self.utility_history]
                
                # Compute correlation
                corr = np.corrcoef(u1, u2)[0,1]
                
                # If strong negative correlation, potential conflict
                if corr < -threshold:
                    conflicts.append((g1, g2))
        
        return conflicts 