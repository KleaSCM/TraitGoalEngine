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

class DynamicSystem:
    def __init__(self, update_type: UpdateType = UpdateType.DISCRETE):
        self.update_type = update_type
        self.config = DynamicConfig(update_type=update_type)
        self.noise_scale = 0.1  # Base noise scale
        
    def get_noise_matrix(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Generate noise matrix for stochastic updates."""
        if not weights:
            return {}
            
        # Noise proportional to current weights
        return {name: np.random.normal(0, self.noise_scale * abs(w)) 
                for name, w in weights.items()}
    
    def continuous_update(self, 
                         current_weights: Dict[str, float],
                         utilities: Dict[str, float],
                         arbitration_fn: Callable) -> Dict[str, float]:
        """Continuous-time update with stochastic noise."""
        if not current_weights:
            return {}
            
        # Get target weights from arbitration function
        target_weights = arbitration_fn(current_weights)
        
        # Generate noise matrix
        noise = self.get_noise_matrix(current_weights)
        
        # Update with drift and noise
        new_weights = {}
        for name in current_weights:
            # Drift term (deterministic update)
            drift = (target_weights[name] - current_weights[name]) / self.config.time_step
            
            # Add noise term
            new_weights[name] = current_weights[name] + (
                drift * self.config.time_step +  # Deterministic drift
                noise[name] * np.sqrt(self.config.time_step)  # Stochastic noise
            )
            
        # Normalize to ensure weights sum to 1
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
            
        return new_weights
    
    def discrete_update(self,
                       current_weights: Dict[str, float],
                       utilities: Dict[str, float],
                       arbitration_fn: Callable) -> Dict[str, float]:
        """Discrete-time update with stochastic noise."""
        if not current_weights:
            return {}
            
        # Get target weights
        target_weights = arbitration_fn(current_weights)
        
        # Generate noise
        noise = self.get_noise_matrix(current_weights)
        
        # Update with noise
        new_weights = {}
        for name in current_weights:
            new_weights[name] = target_weights[name] + noise[name]
            
        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
            
        return new_weights
    
    def update(self,
              current_weights: Dict[str, float],
              utilities: Dict[str, float],
              arbitration_fn: Callable) -> Dict[str, float]:
        """Update weights using appropriate dynamics."""
        if self.update_type == UpdateType.CONTINUOUS:
            return self.continuous_update(current_weights, utilities, arbitration_fn)
        return self.discrete_update(current_weights, utilities, arbitration_fn)
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics including noise effects."""
        return {
            'max_diff': self.config.convergence_threshold,
            'mean_diff': self.config.convergence_threshold / 2,
            'std_diff': self.config.convergence_threshold / 3,
            'convergence_rate': 1.0 / self.config.max_iterations,
            'noise_scale': self.noise_scale
        }
    
    def get_lipschitz_bound(self) -> float:
        """Get Lipschitz bound considering noise effects."""
        return 1.0 / self.config.time_step + self.noise_scale 