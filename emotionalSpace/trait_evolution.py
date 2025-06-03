import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class TraitState:
    """Represents the state of traits in the system."""
    traits: np.ndarray  # w ∈ ℝ^n
    coupling_strength: np.ndarray  # C ∈ ℝ^(n×m)
    stability: np.ndarray  # s ∈ [0, 1]^n
    influence: np.ndarray  # I ∈ [0, 1]^n
    learning_rate: np.ndarray  # η ∈ [0, 1]^n
    interaction_matrix: np.ndarray  # I ∈ ℝ^(n×n)

class TraitEvolution:
    """Implements enhanced trait evolution dynamics with improved stability and learning."""
    
    def __init__(self, n_traits: int, n_emotions: int, n_desires: int):
        """Initialize the trait evolution system with enhanced features."""
        self.n_traits = n_traits
        self.n_emotions = n_emotions
        self.n_desires = n_desires
        
        # Initialize base trait weights with small random variations
        self.base_trait_weights = np.ones(n_traits) + np.random.normal(0, 0.1, n_traits)
        self.base_trait_weights = np.clip(self.base_trait_weights, 0.1, 1.0)
        
        # Initialize trait state with enhanced components
        total_coupling_dim = n_emotions + n_desires
        self.state = TraitState(
            traits=np.copy(self.base_trait_weights),  # Start with base weights
            coupling_strength=np.random.normal(0, 0.1, (n_traits, total_coupling_dim)),
            stability=np.ones(n_traits) * 0.8,  # Start with high stability
            influence=np.ones(n_traits) * 0.5,
            learning_rate=np.ones(n_traits) * 0.01,
            interaction_matrix=np.random.normal(0, 0.1, (n_traits, n_traits))
        )
        
        # Enhanced evolution parameters
        self.stability_decay = 0.001
        self.influence_growth = 0.002
        self.desire_coupling_rate = 0.05
        self.learning_rate_adaptation = 0.01
        self.interaction_strength = 0.1
        self.stability_threshold = 0.7
        
        # Ensure all matrices are properly initialized
        self.state.coupling_strength = np.clip(self.state.coupling_strength, -0.5, 0.5)
        self.state.interaction_matrix = np.clip(self.state.interaction_matrix, -0.5, 0.5)
        np.fill_diagonal(self.state.interaction_matrix, 0.0)  # No self-interaction
    
    def calculate_trait_metric(self, traits1: np.ndarray, traits2: np.ndarray) -> float:
        """Calculate enhanced trait metric with stability weighting."""
        diff = traits1 - traits2
        stability_weights = np.clip(self.state.stability, 0.1, 1.0)  # Ensure non-zero weights
        return np.sqrt(np.sum(stability_weights * diff**2))
    
    def calculate_trait_norm(self, traits: np.ndarray) -> float:
        """Calculate enhanced trait norm with stability weighting."""
        stability_weights = np.clip(self.state.stability, 0.1, 1.0)  # Ensure non-zero weights
        return np.sqrt(np.sum(stability_weights * traits**2))
    
    def calculate_base_evolution(self, dt: float) -> np.ndarray:
        """Calculate enhanced base evolution with learning rate adaptation."""
        return np.clip(self.state.learning_rate * dt * self.state.traits, -0.1, 0.1)
    
    def calculate_emotional_influence(self, emotions: np.ndarray, dt: float) -> np.ndarray:
        """Calculate enhanced emotional influence with stability modulation."""
        base_influence = self.influence_growth * dt * np.dot(
            self.state.coupling_strength[:, :self.n_emotions],
            emotions
        )
        stability_modulation = np.clip(self.state.stability, 0.1, 1.0)
        return np.clip(base_influence * stability_modulation, -0.1, 0.1)
    
    def calculate_desire_coupling(self, desires: np.ndarray, dt: float) -> np.ndarray:
        """Calculate enhanced desire coupling with interaction effects."""
        if len(desires) != self.n_desires:
            raise ValueError(f"Desires array length {len(desires)} does not match n_desires {self.n_desires}")
        
        base_coupling = self.desire_coupling_rate * dt * np.dot(
            self.state.coupling_strength[:, self.n_emotions:self.n_emotions + self.n_desires],
            desires
        )
        interaction_effect = np.dot(self.state.interaction_matrix, self.state.traits)
        return np.clip(base_coupling * (1 + self.interaction_strength * interaction_effect), -0.1, 0.1)
    
    def calculate_stability(self) -> np.ndarray:
        """Calculate enhanced stability measure with learning adaptation."""
        base_stability = self.state.stability * (1 - self.stability_decay)
        learning_modulation = 1 - np.exp(-np.clip(self.state.learning_rate, 0.001, 0.1))
        return np.clip(base_stability * (1 + learning_modulation), 0.1, 1.0)
    
    def update_learning_rates(self, dt: float) -> None:
        """Update learning rates based on stability and performance."""
        stability_factor = np.clip(self.state.stability, 0.1, 1.0)
        performance_factor = np.clip(np.abs(self.state.traits - self.base_trait_weights), 0.0, 1.0)
        
        # Adapt learning rates based on stability and performance
        self.state.learning_rate += self.learning_rate_adaptation * dt * (
            (1 - stability_factor) * (1 - performance_factor) -  # Increase if unstable and not performing
            stability_factor * performance_factor  # Decrease if stable and performing
        )
        
        # Ensure learning rates stay in reasonable range
        self.state.learning_rate = np.clip(self.state.learning_rate, 0.001, 0.1)
    
    def update_interaction_matrix(self, dt: float) -> None:
        """Update trait interaction matrix based on current states."""
        # Calculate trait differences and their influence
        trait_diffs = np.outer(self.state.traits, self.state.traits)
        stability_weights = np.outer(self.state.stability, self.state.stability)
        
        # Add noise for exploration (increased noise)
        noise = np.random.normal(0, 0.05, self.state.interaction_matrix.shape)
        
        # Update interactions based on trait differences, stability, and noise
        # Increased interaction strength and reduced decay
        self.state.interaction_matrix += dt * (
            0.5 * trait_diffs * stability_weights +  # Increased trait influence
            noise -  # Increased exploration
            self.state.interaction_matrix * 0.01  # Reduced decay term
        )
        
        # Ensure interaction matrix stays bounded and symmetric
        self.state.interaction_matrix = np.clip(self.state.interaction_matrix, -0.5, 0.5)
        self.state.interaction_matrix = (self.state.interaction_matrix + self.state.interaction_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(self.state.interaction_matrix, 0.0)  # No self-interaction
    
    def evolve_traits(self, emotions: np.ndarray, desires: np.ndarray, dt: float) -> None:
        """Evolve the trait system with enhanced dynamics."""
        # Calculate evolution components
        base_evolution = self.calculate_base_evolution(dt)
        emotional_influence = self.calculate_emotional_influence(emotions, dt)
        desire_coupling = self.calculate_desire_coupling(desires, dt)
        
        # Update learning rates and interaction matrix
        self.update_learning_rates(dt)
        self.update_interaction_matrix(dt)
        
        # Calculate interaction effects
        interaction_effect = np.dot(self.state.interaction_matrix, self.state.traits)
        
        # Update traits with all influences
        new_traits = self.state.traits + (
            base_evolution +
            emotional_influence +
            desire_coupling +
            self.interaction_strength * interaction_effect
        )
        
        # Ensure traits stay in reasonable range
        self.state.traits = np.clip(new_traits, 0.1, 1.0)
        
        # Update stability
        self.state.stability = self.calculate_stability()
    
    def get_trait_state(self) -> TraitState:
        """Get the current state of the trait system."""
        return self.state