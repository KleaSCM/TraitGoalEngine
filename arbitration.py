from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

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
    def __init__(self, config: ArbitrationConfig):
        self.config = config
        self.utility_history: List[Dict[str, float]] = []
        self.weight_history: List[Dict[str, float]] = []
        
    def weighted_arbitration(self, utilities: Dict[str, float]) -> Dict[str, float]:
        """
        Weighted Utility Arbitration from Definition 10.
        A(u) = (α1u1, α2u2, ..., αnun) / Σ(αjuj)
        """
        if not self.config.weights:
            # Default to uniform weights if none provided
            weights = {name: 1.0/len(utilities) for name in utilities}
        else:
            weights = self.config.weights
            
        weighted_utils = {name: weights[name] * util 
                         for name, util in utilities.items()}
        total = sum(weighted_utils.values())
        
        if total == 0:
            return {name: 1.0/len(utilities) for name in utilities}
            
        return {name: util/total for name, util in weighted_utils.items()}
    
    def softmax_arbitration(self, utilities: Dict[str, float]) -> Dict[str, float]:
        """
        Softmax Arbitration from Definition 11.
        Ai(u) = exp(ui/τ) / Σ(exp(uj/τ))
        """
        if not utilities:
            return {}
            
        # Scale utilities for numerical stability
        max_util = max(utilities.values())
        exps = {name: np.exp((util - max_util) / self.config.temperature)
                for name, util in utilities.items()}
        total = sum(exps.values())
        
        return {name: exp/total for name, exp in exps.items()}
    
    def compute_nash_equilibrium(self, utilities: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute Nash Equilibrium from Definition 13.
        Uses iterative best response to find equilibrium strategies.
        """
        n_goals = len(utilities)
        if n_goals == 0:
            return {}
            
        # Initialize with uniform weights
        weights = {name: 1.0/n_goals for name in utilities}
        
        for _ in range(self.config.max_iterations):
            old_weights = weights.copy()
            
            # Best response for each goal
            for goal_name in utilities:
                # Compute utility given other goals' weights
                other_weights = {name: w for name, w in weights.items() 
                               if name != goal_name}
                utility = self._compute_goal_utility(goal_name, utilities[goal_name], 
                                                  other_weights)
                
                # Update weight based on utility
                weights[goal_name] = max(0.0, min(1.0, utility))
            
            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                weights = {name: w/total for name, w in weights.items()}
            
            # Check convergence
            if self._check_convergence(old_weights, weights):
                break
                
        return weights
    
    def _compute_goal_utility(self, goal_name: str, 
                            utility_fn: Dict[str, float],
                            other_weights: Dict[str, float]) -> float:
        """Compute utility for a goal given other goals' weights"""
        utility = 0.0
        for other_goal, weight in other_weights.items():
            if other_goal in utility_fn:
                utility += weight * utility_fn[other_goal]
        return utility
    
    def _check_convergence(self, old_weights: Dict[str, float],
                          new_weights: Dict[str, float]) -> bool:
        """Check if weights have converged"""
        return all(abs(old_weights[name] - new_weights[name]) < self.config.convergence_threshold
                  for name in old_weights)
    
    def arbitrate(self, utilities: Dict[str, float]) -> Dict[str, float]:
        """
        Main arbitration method that selects appropriate mechanism based on config.
        """
        if self.config.type == ArbitrationType.WEIGHTED:
            weights = self.weighted_arbitration(utilities)
        elif self.config.type == ArbitrationType.SOFTMAX:
            weights = self.softmax_arbitration(utilities)
        elif self.config.type == ArbitrationType.NASH:
            # Convert utilities to game matrix format
            game_utilities = {name: {other: util for other, util in utilities.items()}
                            for name in utilities}
            weights = self.compute_nash_equilibrium(game_utilities)
        else:
            raise ValueError(f"Unknown arbitration type: {self.config.type}")
            
        # Record history
        self.utility_history.append(utilities)
        self.weight_history.append(weights)
        
        return weights
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """
        Get stability metrics for the arbitration process.
        """
        if len(self.utility_history) < 2:
            return {
                'max_diff': 0.0,
                'mean_diff': 0.0,
                'std_diff': 0.0,
                'convergence_rate': 0.0
            }
            
        old = self.utility_history[-2]
        new = self.utility_history[-1]
        
        # Only compare goals that exist in both old and new
        common_goals = set(old.keys()) & set(new.keys())
        if not common_goals:
            return {
                'max_diff': 0.0,
                'mean_diff': 0.0,
                'std_diff': 0.0,
                'convergence_rate': 0.0
            }
            
        diffs = [abs(old[name] - new[name]) for name in common_goals]
        max_diff = max(diffs) if diffs else 0.0
        mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
        std_diff = np.std(diffs) if diffs else 0.0
        
        # Calculate convergence rate
        if len(self.utility_history) > 2:
            prev_diff = sum(abs(self.utility_history[-3][name] - old[name]) 
                          for name in set(self.utility_history[-3].keys()) & set(old.keys()))
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
        """
        Compute Lipschitz constant bound from Section 5.
        Returns upper bound on the Lipschitz constant of the arbitration operator.
        """
        if self.config.type == ArbitrationType.SOFTMAX:
            return 1.0 / (4.0 * self.config.temperature)
        elif self.config.type == ArbitrationType.WEIGHTED:
            if not self.config.weights:
                return 1.0
            return max(abs(w) for w in self.config.weights.values())
        else:
            # For Nash equilibrium, use empirical bound
            if len(self.weight_history) < 2:
                return float('inf')
            diffs = []
            for i in range(1, len(self.weight_history)):
                old = self.weight_history[i-1]
                new = self.weight_history[i]
                diff = sum(abs(old[name] - new[name]) for name in old)
                diffs.append(diff)
            return max(diffs) if diffs else float('inf') 