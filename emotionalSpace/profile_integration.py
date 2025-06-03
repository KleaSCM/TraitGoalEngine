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