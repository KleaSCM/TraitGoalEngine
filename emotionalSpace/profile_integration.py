import numpy as np
from typing import Dict, List, Tuple
from .personal_profile import PersonalProfile, PROFILE
from .emotional_field import EmotionalField, EmotionalComponent
from .trait_evolution import TraitEvolution, TraitState
from .desire_formation import DesireFormation, DesireState

class ProfileIntegration:
    """Integrates the emotional-trait-desire system with the personal profile."""
    
    def __init__(self, profile):
        """Initialize the profile integration system."""
        self.profile = profile
        
        # Get trait weights from profile
        self.trait_weights = profile.get_trait_weights()
        
        # Get dimensions
        self.n_traits = len(self.trait_weights)
        self.n_emotions = len(profile.get_emotion_trait_correlations())
        self.n_desires = len(profile.desires)  # Base desires
        
        # Initialize emotional field
        self.emotional_field = EmotionalField(
            n_emotions=2,  # Only using positive and negative emotions
            n_traits=self.n_traits
        )
        
        # Initialize trait evolution with correct number of desires
        self.trait_evolution = TraitEvolution(
            n_traits=self.n_traits,
            n_emotions=2,  # Only using positive and negative emotions
            n_desires=self.n_desires  # Use actual number of desires
        )
        
        # Initialize desire formation with profile desires
        self.desire_formation = DesireFormation(profile.desires)
        
        # Initialize history tracking
        self.state_history = []
        self.metric_history = []
    
    def get_system_metrics(self) -> dict:
        """Get comprehensive system metrics."""
        # Calculate stability
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
        
        # Calculate performance
        trait_coherence = np.mean(self.trait_evolution.state.stability)
        emotion_coherence = np.exp(-np.abs(
            self.emotional_field.positive.intensity -
            self.emotional_field.negative.intensity
        ))
        desire_coherence = np.mean(self.desire_formation.state.coherence)
        
        # Calculate system norm
        trait_norm = np.linalg.norm(self.trait_evolution.state.traits)
        emotion_norm = np.sqrt(
            self.emotional_field.positive.intensity**2 +
            self.emotional_field.negative.intensity**2
        )
        desire_norm = np.linalg.norm(self.desire_formation.state.desires)
        
        # Calculate phase space metrics
        if len(self.state_history) >= 2:
            # Use more recent states for better metrics
            recent_states = self.state_history[-min(20, len(self.state_history)):]
            
            # Calculate volumes using standard deviations
            trait_volume = np.prod(np.std([s['traits'] for s in recent_states], axis=0))
            emotion_volume = np.prod(np.std([s['emotions'] for s in recent_states], axis=0))
            desire_volume = np.prod(np.std([s['desires'] for s in recent_states], axis=0))
            
            # Calculate attractor strengths using distances from current state
            current_state = recent_states[-1]
            trait_attractor = np.mean([
                np.linalg.norm(s['traits'] - current_state['traits'])
                for s in recent_states[:-1]
            ])
            emotion_attractor = np.mean([
                np.linalg.norm(s['emotions'] - current_state['emotions'])
                for s in recent_states[:-1]
            ])
            desire_attractor = np.mean([
                np.linalg.norm(s['desires'] - current_state['desires'])
                for s in recent_states[:-1]
            ])
        else:
            # Initialize with small non-zero values
            trait_volume = emotion_volume = desire_volume = 0.001
            trait_attractor = emotion_attractor = desire_attractor = 0.001
        
        return {
            'stability': (trait_stability + emotion_stability + desire_stability) / 3,
            'performance': (trait_coherence + emotion_coherence + desire_coherence) / 3,
            'system_norm': np.sqrt(trait_norm**2 + emotion_norm**2 + desire_norm**2),
            'trait_volume': trait_volume,
            'emotion_volume': emotion_volume,
            'desire_volume': desire_volume,
            'trait_attractor': trait_attractor,
            'emotion_attractor': emotion_attractor,
            'desire_attractor': desire_attractor
        }
    
    def evolve_with_profile(self, dt: float) -> None:
        """Evolve the system with profile influence."""
        # Get current states
        trait_state = self.trait_evolution.get_trait_state().traits
        positive_emotion, negative_emotion = self.emotional_field.get_field_state()
        emotion_state = np.array([positive_emotion.intensity, negative_emotion.intensity])
        desire_states = self.desire_formation.get_active_desires()
        
        # Initialize desire state with correct size
        desire_state = np.zeros(self.n_desires)  # Use actual number of desires
        if desire_states:
            for i, d in enumerate(desire_states):
                if i < self.n_desires:  # Ensure we don't exceed array bounds
                    desire_state[i] = d.intensity
        
        # Evolve traits with profile influence
        self.trait_evolution.evolve_traits(emotion_state, desire_state, dt)
        
        # Evolve emotions with profile influence
        self.emotional_field.evolve_field(trait_state, dt)
        
        # Evolve desires with profile influence
        self.desire_formation.evolve(emotion_state, trait_state, dt)
        
        # Store state and metrics
        current_state = {
            'traits': trait_state,
            'emotions': emotion_state,
            'desires': desire_state
        }
        self.state_history.append(current_state)
        self.metric_history.append(self.get_system_metrics())
    
    def get_profile_influenced_state(self) -> Tuple[np.ndarray, np.ndarray, List[DesireState]]:
        """Get the current state with profile influence."""
        trait_state = self.trait_evolution.get_trait_state().traits
        positive_emotion, negative_emotion = self.emotional_field.get_field_state()
        emotion_state = np.array([positive_emotion.intensity, negative_emotion.intensity])
        desire_states = self.desire_formation.get_active_desires()
        
        return trait_state, emotion_state, desire_states
    
    def get_profile_metrics(self) -> dict:
        """Calculate profile-based metrics."""
        trait_state, emotion_state, desire_states = self.get_profile_influenced_state()
        
        # Calculate trait alignment
        trait_alignment = np.mean([
            abs(trait_state[i] - weight)
            for i, (_, weight) in enumerate(self.trait_weights.items())
        ])
        
        # Calculate emotion alignment
        emotion_correlations = self.profile.get_emotion_trait_correlations()
        emotion_alignment = np.mean([
            abs(emotion_state[i])
            for i in range(len(emotion_state))  # Only iterate over our 2 emotions
        ])
        
        # Calculate desire alignment
        if desire_states:
            desire_alignment = np.mean([
                d.intensity / (d.base_desire.importance * d.base_desire.frequency)
                for d in desire_states
            ])
        else:
            desire_alignment = 0.0
        
        # Calculate overall alignment
        overall_alignment = (trait_alignment + emotion_alignment + desire_alignment) / 3
        
        return {
            'trait_alignment': trait_alignment,
            'emotion_alignment': emotion_alignment,
            'desire_alignment': desire_alignment,
            'overall_alignment': overall_alignment
        }