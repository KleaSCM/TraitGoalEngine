import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

@dataclass
class EmotionalComponent:
    """Represents a single emotional component with valence, arousal, dominance, and intensity."""
    valence: float  # v ∈ [-1, 1]
    arousal: float  # a ∈ [0, 1]
    dominance: float  # d ∈ [0, 1]
    intensity: float  # i ∈ [0, 1]
    field_potential: float  # φ
    memory: float = 0.0  # Emotional memory component
    resonance: float = 0.0  # Resonance with other emotions

    def __post_init__(self):
        """Validate component values are within bounds."""
        self.valence = np.clip(self.valence, -1.0, 1.0)
        self.arousal = np.clip(self.arousal, 0.0, 1.0)
        self.dominance = np.clip(self.dominance, 0.0, 1.0)
        self.intensity = np.clip(self.intensity, 0.0, 1.0)
        self.memory = np.clip(self.memory, 0.0, 1.0)
        self.resonance = np.clip(self.resonance, 0.0, 1.0)

class EmotionalField:
    """Implements the enhanced emotional field model with memory and resonance."""
    
    def __init__(self, n_traits: int, n_emotions: int):
        """
        Initialize the emotional field.
        
        Args:
            n_traits: Number of trait dimensions
            n_emotions: Number of emotion components
        """
        self.n_traits = n_traits
        self.n_emotions = n_emotions
        
        # Initialize positive and negative components with memory
        self.positive = EmotionalComponent(0.0, 0.0, 0.0, 0.0, 0.0)
        self.negative = EmotionalComponent(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Initialize field weights and basis functions
        self.weights = np.random.randn(n_emotions)
        self.basis_functions = self._initialize_basis_functions()
        
        # Enhanced field parameters
        self.diffusion_coefficient = 0.2
        self.decay_rate = 0.15
        self.memory_decay = 0.08
        self.resonance_strength = 0.15
        self.coupling_strength = 0.1
        
        # Initialize memory and resonance matrices
        self.emotional_memory = np.zeros((n_emotions, n_traits))
        self.resonance_matrix = np.eye(n_emotions) * 0.5
        
    def _initialize_basis_functions(self) -> List[callable]:
        """Initialize enhanced basis functions for field potential calculation."""
        # Using scaled Gaussian basis functions with memory influence
        centers = np.random.randn(self.n_emotions, self.n_traits) * 0.1
        return [lambda x, center=center: np.exp(-0.5 * np.sum((x - center)**2))
                for center in centers]
    
    def calculate_field_potential(self, trait_state: np.ndarray) -> float:
        """
        Calculate the enhanced field potential φ with memory and resonance.
        
        Args:
            trait_state: Current state of traits
            
        Returns:
            Field potential value
        """
        # Normalize trait state
        trait_state = np.clip(trait_state, -1.0, 1.0)
        
        # Calculate base potential
        base_potential = 0.0
        for i, (weight, basis) in enumerate(zip(self.weights, self.basis_functions)):
            base_potential += weight * basis(trait_state)
        
        # Add memory influence
        memory_potential = np.sum(self.emotional_memory * trait_state.reshape(1, -1))
        
        # Add resonance influence
        resonance_potential = np.sum(self.resonance_matrix * np.outer(self.weights, self.weights))
        
        # Combine potentials with appropriate weights
        total_potential = (base_potential + 
                          self.memory_decay * memory_potential + 
                          self.resonance_strength * resonance_potential)
        
        return np.clip(total_potential, -1.0, 1.0)
    
    def calculate_field_strength(self) -> float:
        """
        Calculate the enhanced field strength including memory and resonance.
        
        Returns:
            Field strength value
        """
        # Calculate base field strength
        positive_norm = np.sqrt(self.positive.valence**2 + self.positive.arousal**2 + 
                              self.positive.dominance**2 + self.positive.intensity**2)
        negative_norm = np.sqrt(self.negative.valence**2 + self.negative.arousal**2 + 
                              self.negative.dominance**2 + self.negative.intensity**2)
        
        # Add memory and resonance contributions
        memory_strength = np.mean(np.abs(self.emotional_memory))
        resonance_strength = np.mean(np.abs(self.resonance_matrix))
        
        return np.sqrt(positive_norm**2 + negative_norm**2 + 
                      self.memory_decay * memory_strength**2 + 
                      self.resonance_strength * resonance_strength**2)
    
    def calculate_field_gradient(self, trait_state: np.ndarray) -> np.ndarray:
        """
        Calculate the enhanced gradient of the field potential.
        
        Args:
            trait_state: Current state of traits
            
        Returns:
            Gradient vector
        """
        gradient = np.zeros(self.n_traits)
        
        # Calculate base gradient
        for i, (weight, basis) in enumerate(zip(self.weights, self.basis_functions)):
            epsilon = 1e-6
            for j in range(self.n_traits):
                trait_plus = trait_state.copy()
                trait_plus[j] += epsilon
                trait_minus = trait_state.copy()
                trait_minus[j] -= epsilon
                gradient[j] += weight * (basis(trait_plus) - basis(trait_minus)) / (2 * epsilon)
        
        # Add memory gradient
        memory_gradient = np.sum(self.emotional_memory, axis=0)
        
        # Add resonance gradient
        resonance_gradient = np.sum(self.resonance_matrix * self.weights.reshape(-1, 1), axis=0)
        
        return gradient + self.memory_decay * memory_gradient + self.resonance_strength * resonance_gradient
    
    def evolve_field(self, trait_state: np.ndarray, dt: float) -> None:
        """Evolve the emotional field with enhanced dynamics."""
        # Calculate emotional memory influence
        memory_influence = np.mean(self.emotional_memory) * 0.3  # Increased from 0.2
        
        # Calculate trait influence with stronger coupling
        trait_influence = np.mean(trait_state) * 0.4  # Increased from 0.3
        
        # Add stronger random noise
        noise = np.random.normal(0, 0.3)  # Increased from 0.2
        
        # Update positive emotion with stronger feedback
        positive_update = (
            memory_influence * (1 + noise) +
            trait_influence * (1 + 0.2 * noise) -  # Added noise to trait influence
            self.decay_rate * self.positive.intensity * 1.2  # Increased decay
        )
        self.positive.intensity = np.clip(
            self.positive.intensity + dt * positive_update,
            0.0, 1.0
        )
        
        # Update negative emotion with stronger feedback
        negative_update = (
            -memory_influence * (1 + noise) -
            trait_influence * (1 + 0.2 * noise) +  # Added noise to trait influence
            self.decay_rate * self.negative.intensity * 1.2  # Increased decay
        )
        self.negative.intensity = np.clip(
            self.negative.intensity + dt * negative_update,
            0.0, 1.0
        )
        
        # Update emotional memory with stronger decay
        self.emotional_memory = (self.emotional_memory * (1 - self.memory_decay * dt * 1.5) +  # Increased decay
            dt * np.outer(np.array([self.positive.intensity, self.negative.intensity]), np.ones(self.n_traits)))
        
        # Update resonance matrix with stronger random updates
        self.resonance_matrix += dt * np.random.normal(0, 0.3, self.resonance_matrix.shape)  # Increased noise
        self.resonance_matrix = np.clip(self.resonance_matrix, -1.0, 1.0)
        
        # Apply diffusion and decay with enhanced rates
        decay_rate = self.decay_rate * (1 + np.abs(trait_influence) + memory_influence)
        self.weights *= (1 - decay_rate * dt)
        
        # Add larger random updates to weights
        self.weights += np.random.normal(0, 0.2, size=self.weights.shape)  # Increased from 0.05
        
    def get_field_state(self) -> Tuple[EmotionalComponent, EmotionalComponent]:
        """
        Get the current state of the emotional field.
        
        Returns:
            Tuple of (positive_component, negative_component)
        """
        return self.positive, self.negative 