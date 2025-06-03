import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class TraitState:
    """Represents the state of traits in the system."""
    traits: np.ndarray  # w ∈ ℝ^n
    coupling_strength: np.ndarray  # C ∈ ℝ^(n×m)
    stability: np.ndarray  # s ∈ [0, 1]^n
    influence: np.ndarray  # I ∈ [0, 1]^n

class TraitEvolution:
    """Implements the trait evolution dynamics as defined in section 28 of the framework."""
    
    def __init__(self, n_traits: int, n_emotions: int, n_desires: int):
        """
        Initialize the trait evolution system.
        
        Args:
            n_traits: Number of trait dimensions
            n_emotions: Number of emotion dimensions
            n_desires: Number of desire dimensions
        """
        self.n_traits = n_traits
        self.n_emotions = n_emotions
        self.n_desires = n_desires
        
        # Initialize base trait weights (these will never decrease)
        self.base_trait_weights = np.ones(n_traits)  # All traits start at base weight 1.0
        
        # Initialize trait state with correct dimensions
        total_coupling_dim = n_emotions + n_desires  # Total dimensions for emotions + desires
        self.state = TraitState(
            traits=np.ones(n_traits),  # Start with neutral traits
            coupling_strength=np.random.randn(n_traits, total_coupling_dim) * 0.1,  # Correct dimensions
            stability=np.ones(n_traits) * 0.8,  # High initial stability
            influence=np.ones(n_traits) * 0.5   # Moderate initial influence
        )
        
        # Evolution parameters
        self.learning_rate = 0.01
        self.stability_decay = 0.001
        self.influence_growth = 0.002
        self.desire_coupling_rate = 0.05  # Rate at which desires influence traits
    
    def calculate_trait_metric(self, traits1: np.ndarray, traits2: np.ndarray) -> float:
        """
        Calculate the trait metric as defined in section 28.1.
        
        Args:
            traits1: First trait vector
            traits2: Second trait vector
            
        Returns:
            Metric distance between traits
        """
        diff = traits1 - traits2
        return np.sqrt(np.sum(diff**2))
    
    def calculate_trait_norm(self, traits: np.ndarray) -> float:
        """
        Calculate the trait norm as defined in section 28.1.
        
        Args:
            traits: Trait vector
            
        Returns:
            Norm of the trait vector
        """
        return np.sqrt(np.sum(traits**2))
    
    def calculate_base_evolution(self, dt: float) -> np.ndarray:
        """
        Calculate the base evolution of traits as defined in section 28.2.
        
        Args:
            dt: Time step
            
        Returns:
            Evolution vector
        """
        return self.learning_rate * dt * self.state.traits
    
    def calculate_emotional_influence(self, emotions: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculate the emotional influence on traits as defined in section 28.2.
        
        Args:
            emotions: Current emotional state
            dt: Time step
            
        Returns:
            Emotional influence vector
        """
        return self.influence_growth * dt * np.dot(
            self.state.coupling_strength[:, :self.n_emotions],
            emotions
        )
    
    def calculate_desire_coupling(self, desires: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculate the desire coupling effect as defined in section 28.2.
        
        Args:
            desires: Current desire state
            dt: Time step
            
        Returns:
            Desire coupling vector
        """
        # Ensure desires array has the correct length
        if len(desires) != self.n_desires:
            raise ValueError(f"Desires array length {len(desires)} does not match n_desires {self.n_desires}")
            
        return self.desire_coupling_rate * dt * np.dot(
            self.state.coupling_strength[:, self.n_emotions:self.n_emotions + self.n_desires],
            desires
        )
    
    def calculate_stability(self) -> np.ndarray:
        """
        Calculate the stability measure for each trait.
        
        Returns:
            Array of stability measures
        """
        # Calculate stability based on trait values and influence
        stability = self.state.stability * (1 - self.stability_decay)
        stability = np.clip(stability, 0.0, 1.0)
        return stability
    
    def evolve_traits(self, emotions: np.ndarray, desires: np.ndarray, dt: float) -> None:
        """
        Evolve the trait system according to section 28.2.
        
        Args:
            emotions: Current emotional state
            desires: Current desire state
            dt: Time step
        """
        # Calculate evolution components with enhanced feedback
        base_evolution = self.calculate_base_evolution(dt)
        
        # Emotional influence with desire modulation
        emotional_influence = self.calculate_emotional_influence(emotions, dt)
        desire_modulation = np.mean(desires) if len(desires) > 0 else 0.0
        emotional_influence *= (1 + desire_modulation)  # Desires amplify emotional influence
        
        # Desire coupling with emotional modulation
        desire_coupling = self.calculate_desire_coupling(desires, dt)
        emotion_modulation = np.mean(np.abs(emotions))
        desire_coupling *= (1 + emotion_modulation)  # Emotions amplify desire influence
        
        # Update traits with feedback, ensuring they never go below base weights
        new_traits = self.state.traits + base_evolution + emotional_influence + desire_coupling
        self.state.traits = np.maximum(new_traits, self.base_trait_weights)
        
        # Update stability
        self.state.stability = self.calculate_stability()
    
    def get_trait_state(self) -> TraitState:
        """
        Get the current state of the trait system.
        
        Returns:
            Current trait state
        """
        return self.state