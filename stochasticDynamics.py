from typing import Dict, Callable, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class SDEConfig:
    """Configuration for stochastic differential equations"""
    time_step: float = 0.01  # Time step for numerical integration
    noise_scale: float = 0.1  # Base noise scale
    mean_reversion_rate: float = 0.1  # Mean reversion rate
    stability_factor: float = 1.0  # Stability scaling factor
    base_temperature: float = 1.0  # Base temperature for softmax
    resilience_weight: float = 0.5  # Weight for resilience trait influence
    decisiveness_weight: float = 0.5  # Weight for decisiveness trait influence

class StochasticDynamics:
    """
    Implementation of continuous-time stochastic processes for utility and weight evolution.
    Uses Euler-Maruyama method for numerical integration of SDEs.
    """
    def __init__(self, config: Optional[SDEConfig] = None):
        self.config = config or SDEConfig()
    
    def get_effective_temperature(self, trait_values: Dict[str, float]) -> float:
        """
        Calculate effective temperature based on trait states.
        Higher resilience and decisiveness lead to lower temperature (more stable).
        """
        # Get trait values with defaults
        resilience = trait_values.get('resilience', 0.5)
        decisiveness = trait_values.get('decisiveness', 0.5)
        
        # Calculate trait influence
        resilience_influence = self.config.resilience_weight * (1 - resilience)  # Inverse relationship
        decisiveness_influence = self.config.decisiveness_weight * (1 - decisiveness)  # Inverse relationship
        
        # Combine influences
        trait_modulation = resilience_influence + decisiveness_influence
        
        # Scale base temperature by trait modulation
        effective_temp = self.config.base_temperature * (1 + trait_modulation)
        
        return max(0.1, min(2.0, effective_temp))  # Bound temperature between 0.1 and 2.0
    
    def utility_sde(self, 
                   current_utility: float,
                   target_utility: float,
                   stability: float,
                   temperature: float,
                   trait_values: Optional[Dict[str, float]] = None) -> float:
        """
        Stochastic differential equation for utility evolution:
        dU = α(μ - U)dt + σ√dt dW
        
        where:
        - U is the utility
        - α is the mean reversion rate (scaled by stability and traits)
        - μ is the target utility
        - σ is the noise scale (scaled by temperature, stability, and traits)
        - dW is the Wiener process increment
        """
        if trait_values is None:
            trait_values = {}
        
        # Get effective temperature based on traits
        effective_temp = self.get_effective_temperature(trait_values)
        
        # Scale mean reversion by stability and resilience
        resilience = trait_values.get('resilience', 0.5)
        alpha = self.config.mean_reversion_rate * stability * (1 + resilience)
        
        # Scale noise by temperature, stability, and decisiveness
        decisiveness = trait_values.get('decisiveness', 0.5)
        sigma = self.config.noise_scale * effective_temp * (1 - stability) * (1 - decisiveness)
        
        # Drift term (deterministic)
        drift = alpha * (target_utility - current_utility)
        
        # Diffusion term (stochastic)
        diffusion = sigma * np.sqrt(self.config.time_step) * np.random.normal()
        
        # Euler-Maruyama update
        new_utility = current_utility + drift * self.config.time_step + diffusion
        
        # Ensure utility stays in [0,1]
        return max(0.0, min(1.0, new_utility))
    
    def weight_sde(self,
                  current_weights: Dict[str, float],
                  target_weights: Dict[str, float],
                  utilities: Dict[str, float],
                  temperature: float,
                  trait_values: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Stochastic differential equation for weight evolution:
        dW = β(U - W)dt + γ√dt dW
        
        where:
        - W is the weight vector
        - β is the adaptation rate (scaled by traits)
        - U is the utility vector
        - γ is the noise scale (scaled by temperature and traits)
        - dW is the Wiener process increment
        """
        if trait_values is None:
            trait_values = {}
        
        # Get effective temperature based on traits
        effective_temp = self.get_effective_temperature(trait_values)
        
        new_weights = {}
        total_utility = sum(utilities.values())
        
        if total_utility > 0:
            for name in current_weights:
                # Scale adaptation rate by utility and decisiveness
                decisiveness = trait_values.get('decisiveness', 0.5)
                beta = self.config.mean_reversion_rate * (utilities[name] / total_utility) * (1 + decisiveness)
                
                # Scale noise by temperature, utility, and resilience
                resilience = trait_values.get('resilience', 0.5)
                gamma = self.config.noise_scale * effective_temp * (1 - utilities[name] / total_utility) * (1 - resilience)
                
                # Drift term
                drift = beta * (target_weights[name] - current_weights[name])
                
                # Diffusion term
                diffusion = gamma * np.sqrt(self.config.time_step) * np.random.normal()
                
                # Euler-Maruyama update
                new_weights[name] = current_weights[name] + drift * self.config.time_step + diffusion
        
        # Normalize weights
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        return new_weights
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics for the stochastic processes."""
        return {
            'mean_reversion_rate': self.config.mean_reversion_rate,
            'noise_scale': self.config.noise_scale,
            'time_step': self.config.time_step,
            'base_temperature': self.config.base_temperature,
            'resilience_weight': self.config.resilience_weight,
            'decisiveness_weight': self.config.decisiveness_weight
        } 