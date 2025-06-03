import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from emotionalSpace.emotional_field import EmotionalField, EmotionalComponent
from emotionalSpace.trait_evolution import TraitEvolution, TraitState
from emotionalSpace.desire_formation import DesireFormation, DesireState

@dataclass
class SystemState:
    """Represents the complete system state."""
    traits: TraitState
    emotions: Tuple[EmotionalComponent, EmotionalComponent]
    desires: DesireState
    stability: float
    performance: float

class SystemIntegration:
    """Implements the system integration as defined in section 30 of the framework."""
    
    def __init__(self, n_traits: int, n_emotions: int, n_desires: int):
        """
        Initialize the integrated system.
        
        Args:
            n_traits: Number of trait dimensions
            n_emotions: Number of emotion dimensions
            n_desires: Number of desire dimensions
        """
        self.n_traits = n_traits
        self.n_emotions = n_emotions
        self.n_desires = n_desires
        
        # Initialize subsystems
        self.emotional_field = EmotionalField(n_traits, n_emotions)
        self.trait_evolution = TraitEvolution(n_traits, n_emotions, n_desires)
        self.desire_formation = DesireFormation(n_desires, n_traits, n_emotions)
        
        # Integration parameters
        self.coupling_strength = 0.1
        self.stability_threshold = 0.5
        self.performance_threshold = 0.7
        
    def calculate_system_metric(self, state1: SystemState, state2: SystemState) -> float:
        """
        Calculate the system metric as defined in section 30.1.
        
        Args:
            state1: First system state
            state2: Second system state
            
        Returns:
            Metric distance between states
        """
        trait_distance = self.trait_evolution.calculate_trait_metric(
            state1.traits.traits,
            state2.traits.traits
        )
        
        emotion_distance = np.sqrt(
            np.sum((state1.emotions[0].intensity - state2.emotions[0].intensity)**2) +
            np.sum((state1.emotions[1].intensity - state2.emotions[1].intensity)**2)
        )
        
        desire_distance = self.desire_formation.calculate_desire_metric(
            state1.desires.desires,
            state2.desires.desires
        )
        
        return np.sqrt(
            trait_distance**2 +
            emotion_distance**2 +
            desire_distance**2
        )
    
    def calculate_system_norm(self, state: SystemState) -> float:
        """
        Calculate the system norm as defined in section 30.1.
        
        Args:
            state: System state
            
        Returns:
            Norm of the system state
        """
        trait_norm = self.trait_evolution.calculate_trait_norm(state.traits.traits)
        emotion_norm = np.sqrt(
            np.sum(state.emotions[0].intensity**2) +
            np.sum(state.emotions[1].intensity**2)
        )
        desire_norm = self.desire_formation.calculate_desire_norm(state.desires.desires)
        
        return np.sqrt(
            trait_norm**2 +
            emotion_norm**2 +
            desire_norm**2
        )
    
    def calculate_stability(self) -> float:
        """
        Calculate the system stability as defined in section 30.3.
        
        Returns:
            Stability measure
        """
        trait_stability = self.trait_evolution.state.stability
        emotion_stability = np.mean([
            self.emotional_field.positive.intensity,
            self.emotional_field.negative.intensity
        ])
        desire_stability = self.desire_formation.state.stability
        
        return np.clip(
            (trait_stability + emotion_stability + desire_stability) / 3,
            0.0, 1.0
        )
    
    def calculate_performance(self) -> float:
        """
        Calculate the system performance as defined in section 30.6.
        
        Returns:
            Performance measure
        """
        # Calculate coherence
        trait_coherence = self.trait_evolution.state.stability
        emotion_coherence = np.exp(-np.abs(
            self.emotional_field.positive.intensity -
            self.emotional_field.negative.intensity
        ))
        desire_coherence = self.desire_formation.state.coherence
        
        # Calculate efficiency
        trait_efficiency = np.exp(-np.mean(np.abs(
            self.trait_evolution.state.plasticity
        )))
        emotion_efficiency = np.exp(-np.mean(np.abs(
            self.emotional_field.weights
        )))
        desire_efficiency = np.exp(-np.mean(np.abs(
            self.desire_formation.state.flow_rates
        )))
        
        return np.clip(
            (trait_coherence + emotion_coherence + desire_coherence +
             trait_efficiency + emotion_efficiency + desire_efficiency) / 6,
            0.0, 1.0
        )
    
    def evolve_system(self, dt: float) -> None:
        """
        Evolve the integrated system according to section 30.2.
        
        Args:
            dt: Time step
        """
        # Get current states
        trait_state = self.trait_evolution.get_trait_state()
        emotion_state = self.emotional_field.get_field_state()
        desire_state = self.desire_formation.get_desire_state()
        
        # Evolve emotional field
        self.emotional_field.evolve_field(trait_state.traits, dt)
        
        # Evolve trait system
        self.trait_evolution.evolve_traits(
            np.array([emotion_state[0].intensity, emotion_state[1].intensity]),
            desire_state.desires,
            dt
        )
        
        # Evolve desire system
        self.desire_formation.evolve_desires(
            trait_state.traits,
            np.array([emotion_state[0].intensity, emotion_state[1].intensity]),
            dt
        )
    
    def get_system_state(self) -> SystemState:
        """
        Get the current state of the integrated system.
        
        Returns:
            Current system state
        """
        return SystemState(
            traits=self.trait_evolution.get_trait_state(),
            emotions=self.emotional_field.get_field_state(),
            desires=self.desire_formation.get_desire_state(),
            stability=self.calculate_stability(),
            performance=self.calculate_performance()
        ) 