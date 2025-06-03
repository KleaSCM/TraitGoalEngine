from typing import Dict, List, Callable, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class UpdateType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"

@dataclass
class DynamicConfig:
    """Configuration for dynamic update rules"""
    update_type: UpdateType
    time_step: float = 0.1  # For continuous updates
    max_iterations: int = 100
    convergence_threshold: float = 1e-6

class DynamicUpdateSystem:
    """
    Implementation of dynamic update rules from Section 7.1.
    Handles both discrete and continuous-time evolution of arbitration weights.
    """
    def __init__(self, config: DynamicConfig):
        self.config = config
        self.weight_history: List[Dict[str, float]] = []
        self.utility_history: List[Dict[str, float]] = []
        
    def discrete_update(self, 
                       current_weights: Dict[str, float],
                       utilities: Dict[str, float],
                       arbitration_fn: Callable) -> Dict[str, float]:
        """
        Discrete dynamical system update (Equation 8):
        α(t+1) = Φ(α(t), U(t))
        
        Args:
            current_weights: Current arbitration weights α(t)
            utilities: Current goal utilities U(t)
            arbitration_fn: Arbitration rule Φ (e.g., softmax or Nash-based)
            
        Returns:
            Updated weights α(t+1)
        """
        # Record history
        self.weight_history.append(current_weights)
        self.utility_history.append(utilities)
        
        # Apply arbitration rule
        new_weights = arbitration_fn(utilities)
        
        return new_weights
    
    def continuous_update(self,
                         current_weights: Dict[str, float],
                         utilities: Dict[str, float],
                         arbitration_fn: Callable) -> Dict[str, float]:
        """
        Continuous-time update (Equation 9):
        dα(t)/dt = F(α(t), U(t))
        
        Uses Euler integration to approximate the continuous dynamics.
        
        Args:
            current_weights: Current arbitration weights α(t)
            utilities: Current goal utilities U(t)
            arbitration_fn: Arbitration rule Φ
            
        Returns:
            Updated weights after one time step
        """
        # Record history
        self.weight_history.append(current_weights)
        self.utility_history.append(utilities)
        
        # Compute rate of change F(α(t), U(t))
        target_weights = arbitration_fn(utilities)
        rate_of_change = {
            name: (target_weights[name] - current_weights[name]) / self.config.time_step
            for name in current_weights
        }
        
        # Euler integration step
        new_weights = {
            name: current_weights[name] + self.config.time_step * rate_of_change[name]
            for name in current_weights
        }
        
        # Normalize to ensure weights sum to 1
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {name: w/total for name, w in new_weights.items()}
        
        return new_weights
    
    def update(self,
              current_weights: Dict[str, float],
              utilities: Dict[str, float],
              arbitration_fn: Callable) -> Dict[str, float]:
        """
        Main update method that selects appropriate dynamics based on config.
        
        Args:
            current_weights: Current arbitration weights α(t)
            utilities: Current goal utilities U(t)
            arbitration_fn: Arbitration rule Φ
            
        Returns:
            Updated weights α(t+1) or α(t + dt)
        """
        if self.config.update_type == UpdateType.DISCRETE:
            return self.discrete_update(current_weights, utilities, arbitration_fn)
        else:
            return self.continuous_update(current_weights, utilities, arbitration_fn)
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """
        Compute stability metrics for the dynamic update process.
        """
        if len(self.weight_history) < 2:
            return {
                'max_diff': 0.0,
                'mean_diff': 0.0,
                'std_diff': 0.0,
                'convergence_rate': 0.0
            }
        
        # Get last two weight vectors
        old = self.weight_history[-2]
        new = self.weight_history[-1]
        
        # Compute differences
        diffs = [abs(old[name] - new[name]) for name in old]
        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)
        std_diff = np.std(diffs)
        
        # Compute convergence rate
        if len(self.weight_history) > 2:
            prev_diff = sum(abs(self.weight_history[-3][name] - old[name]) 
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
        """
        Compute Lipschitz constant bound for the dynamic update operator.
        """
        if len(self.weight_history) < 2:
            return float('inf')
        
        diffs = []
        for i in range(1, len(self.weight_history)):
            old = self.weight_history[i-1]
            new = self.weight_history[i]
            diff = sum(abs(old[name] - new[name]) for name in old)
            diffs.append(diff)
        
        return max(diffs) if diffs else float('inf') 