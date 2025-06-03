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
        
        # History tracking for stability analysis
        self.state_history = []
        self.metric_history = []
        
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
        trait_stability = np.mean(self.trait_evolution.state.stability)
        emotion_stability = np.mean([
            self.emotional_field.positive.intensity,
            self.emotional_field.negative.intensity
        ])
        desire_stability = np.mean(self.desire_formation.state.stability)
        
        # Calculate coupling stability
        trait_emotion_coupling = np.mean(np.abs(
            self.trait_evolution.state.coupling_strength[:, :self.n_emotions]
        ))
        trait_desire_coupling = np.mean(np.abs(
            self.trait_evolution.state.coupling_strength[:, self.n_emotions:]
        ))
        interaction_stability = np.mean(np.abs(
            self.trait_evolution.state.interaction_matrix
        ))
        
        # Combine stability measures with coupling effects
        base_stability = (trait_stability + emotion_stability + desire_stability) / 3
        coupling_stability = (trait_emotion_coupling + trait_desire_coupling + interaction_stability) / 3
        
        return np.clip(
            base_stability * (1 + self.coupling_strength * coupling_stability),
            0.0, 1.0
        )
    
    def calculate_performance(self) -> float:
        """
        Calculate the system performance as defined in section 30.6.
        
        Returns:
            Performance measure
        """
        # Calculate coherence
        trait_coherence = np.mean(self.trait_evolution.state.stability)
        emotion_coherence = np.exp(-np.abs(
            self.emotional_field.positive.intensity -
            self.emotional_field.negative.intensity
        ))
        desire_coherence = np.mean(self.desire_formation.state.coherence)
        
        # Calculate efficiency
        trait_efficiency = np.exp(-np.mean(np.abs(
            self.trait_evolution.state.learning_rate
        )))
        emotion_efficiency = np.exp(-np.mean(np.abs(
            self.emotional_field.weights
        )))
        desire_efficiency = np.exp(-np.mean(np.abs(
            self.desire_formation.state.flow_rates
        )))
        
        # Calculate integration
        trait_emotion_integration = np.mean(np.abs(
            self.trait_evolution.state.coupling_strength[:, :self.n_emotions]
        ))
        trait_desire_integration = np.mean(np.abs(
            self.trait_evolution.state.coupling_strength[:, self.n_emotions:]
        ))
        interaction_integration = np.mean(np.abs(
            self.trait_evolution.state.interaction_matrix
        ))
        
        # Combine all metrics
        coherence = (trait_coherence + emotion_coherence + desire_coherence) / 3
        efficiency = (trait_efficiency + emotion_efficiency + desire_efficiency) / 3
        integration = (trait_emotion_integration + trait_desire_integration + interaction_integration) / 3
        
        return np.clip(
            (coherence + efficiency + self.coupling_strength * integration) / 3,
            0.0, 1.0
        )
    
    def calculate_phase_space_metrics(self) -> dict:
        """
        Calculate metrics for phase space analysis.
        
        Returns:
            Dictionary of phase space metrics
        """
        if len(self.state_history) < 2:
            return {}
            
        # Calculate phase space volume
        recent_states = self.state_history[-10:]
        trait_volume = np.mean([
            np.prod(np.std([s.traits.traits for s in recent_states], axis=0))
        ])
        emotion_volume = np.mean([
            np.prod(np.std([s.emotions[0].intensity for s in recent_states], axis=0))
        ])
        desire_volume = np.mean([
            np.prod(np.std([s.desires.desires for s in recent_states], axis=0))
        ])
        
        # Calculate attractor strength
        trait_attractor = np.mean([
            np.linalg.norm(s.traits.traits - recent_states[-1].traits.traits)
            for s in recent_states[:-1]
        ])
        emotion_attractor = np.mean([
            np.linalg.norm(s.emotions[0].intensity - recent_states[-1].emotions[0].intensity)
            for s in recent_states[:-1]
        ])
        desire_attractor = np.mean([
            np.linalg.norm(s.desires.desires - recent_states[-1].desires.desires)
            for s in recent_states[:-1]
        ])
        
        return {
            'trait_volume': trait_volume,
            'emotion_volume': emotion_volume,
            'desire_volume': desire_volume,
            'trait_attractor': trait_attractor,
            'emotion_attractor': emotion_attractor,
            'desire_attractor': desire_attractor
        }
    
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
        
        # Store state and metrics
        current_state = SystemState(
            traits=trait_state,
            emotions=emotion_state,
            desires=desire_state,
            stability=self.calculate_stability(),
            performance=self.calculate_performance()
        )
        self.state_history.append(current_state)
        
        # Store metrics
        self.metric_history.append({
            'stability': current_state.stability,
            'performance': current_state.performance,
            **self.calculate_phase_space_metrics()
        })
    
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
    
    def get_system_metrics(self) -> dict:
        """
        Get comprehensive system metrics.
        
        Returns:
            Dictionary of system metrics
        """
        current_state = self.get_system_state()
        phase_metrics = self.calculate_phase_space_metrics()
        
        return {
            'stability': current_state.stability,
            'performance': current_state.performance,
            'system_norm': self.calculate_system_norm(current_state),
            **phase_metrics
        } 