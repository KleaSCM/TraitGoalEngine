import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class EmotionalComponent:
    """Represents a single emotional component with valence, arousal, dominance, and intensity."""
    valence: float  # v ∈ [-1, 1]
    arousal: float  # a ∈ [0, 1]
    dominance: float  # d ∈ [0, 1]
    intensity: float  # i ∈ [0, 1]
    field_potential: float  # φ

    def __post_init__(self):
        """Validate component values are within bounds."""
        self.valence = np.clip(self.valence, -1.0, 1.0)
        self.arousal = np.clip(self.arousal, 0.0, 1.0)
        self.dominance = np.clip(self.dominance, 0.0, 1.0)
        self.intensity = np.clip(self.intensity, 0.0, 1.0)

class EmotionalField:
    """Implements the emotional field model as defined in section 1 of the framework."""
    
    def __init__(self, n_traits: int, n_emotions: int):
        """
        Initialize the emotional field.
        
        Args:
            n_traits: Number of trait dimensions
            n_emotions: Number of emotion components
        """
        self.n_traits = n_traits
        self.n_emotions = n_emotions
        
        # Initialize positive and negative components
        self.positive = EmotionalComponent(0.0, 0.0, 0.0, 0.0, 0.0)
        self.negative = EmotionalComponent(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Initialize field weights and basis functions
        self.weights = np.random.randn(n_emotions)
        self.basis_functions = self._initialize_basis_functions()
        
        # Field parameters
        self.diffusion_coefficient = 0.1
        self.decay_rate = 0.05
        
    def _initialize_basis_functions(self) -> List[callable]:
        """Initialize basis functions for field potential calculation."""
        # Using Gaussian basis functions
        return [lambda x, i=i: np.exp(-0.5 * np.sum((x - np.random.randn(self.n_traits))**2))
                for i in range(self.n_emotions)]
    
    def calculate_field_potential(self, trait_state: np.ndarray) -> float:
        """
        Calculate the field potential φ as defined in section 1.2.
        
        Args:
            trait_state: Current state of traits
            
        Returns:
            Field potential value
        """
        potential = 0.0
        for i, (weight, basis) in enumerate(zip(self.weights, self.basis_functions)):
            potential += weight * basis(trait_state)
        return potential
    
    def calculate_field_strength(self) -> float:
        """
        Calculate the field strength as defined in section 1.2.
        
        Returns:
            Field strength value
        """
        positive_norm = np.sqrt(self.positive.valence**2 + self.positive.arousal**2 + 
                              self.positive.dominance**2 + self.positive.intensity**2)
        negative_norm = np.sqrt(self.negative.valence**2 + self.negative.arousal**2 + 
                              self.negative.dominance**2 + self.negative.intensity**2)
        return np.sqrt(positive_norm**2 + negative_norm**2)
    
    def calculate_field_gradient(self, trait_state: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the field potential with respect to traits.
        
        Args:
            trait_state: Current state of traits
            
        Returns:
            Gradient vector
        """
        gradient = np.zeros(self.n_traits)
        for i, (weight, basis) in enumerate(zip(self.weights, self.basis_functions)):
            # Numerical gradient approximation
            epsilon = 1e-6
            for j in range(self.n_traits):
                trait_plus = trait_state.copy()
                trait_plus[j] += epsilon
                trait_minus = trait_state.copy()
                trait_minus[j] -= epsilon
                gradient[j] += weight * (basis(trait_plus) - basis(trait_minus)) / (2 * epsilon)
        return gradient
    
    def evolve_field(self, trait_state: np.ndarray, dt: float) -> None:
        """
        Evolve the emotional field according to section 1.4.
        
        Args:
            trait_state: Current state of traits
            dt: Time step
        """
        # Calculate field potential
        potential = self.calculate_field_potential(trait_state)
        
        # Update positive component
        self.positive.field_potential = potential
        self.positive.intensity = np.clip(self.positive.intensity + 
                                        dt * (potential - self.positive.intensity), 0.0, 1.0)
        
        # Update negative component
        self.negative.field_potential = -potential
        self.negative.intensity = np.clip(self.negative.intensity + 
                                        dt * (-potential - self.negative.intensity), 0.0, 1.0)
        
        # Apply diffusion and decay
        self.weights *= (1 - self.decay_rate * dt)
        
    def get_field_state(self) -> Tuple[EmotionalComponent, EmotionalComponent]:
        """
        Get the current state of the emotional field.
        
        Returns:
            Tuple of (positive_component, negative_component)
        """
        return self.positive, self.negative 