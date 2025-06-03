import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from .personal_profile import Desire

@dataclass
class DesireState:
    """Represents the state of a desire in the system."""
    base_desire: Desire  # The original desire from the profile
    intensity: float    # Current intensity (0-1)
    age: float = 0.0   # How long this desire has been active
    decay_rate: float = 0.01
    is_active: bool = True

@dataclass
class DesireSystemState:
    """Represents the state of the entire desire formation system."""
    desires: np.ndarray  # Current intensities of all desires
    stability: np.ndarray  # Stability measures for each desire
    coherence: np.ndarray  # Coherence measures for each desire
    flow_rates: np.ndarray  # Flow rates for each desire

class DesireFormation:
    """Models the formation and evolution of desires based on emotional and trait states."""
    
    def __init__(self, profile_desires: List[Desire]):
        """Initialize with desires from the profile."""
        self.base_desires = profile_desires
        self.desires: List[DesireState] = []
        
        # Initialize all profile desires with their base weights
        for desire in profile_desires:
            self.desires.append(DesireState(
                base_desire=desire,
                intensity=desire.importance * desire.frequency,  # Start at base weight
                decay_rate=0.01,
                is_active=True  # Start all desires as active
            ))
        
        # Initialize conflict resolution parameters
        self.conflict_threshold = 0.7  # Threshold for considering desires in conflict
        self.resolution_strength = 0.3  # How strongly to resolve conflicts
        self.emotional_modulation = 0.5  # How much emotions influence conflict resolution
        
        # Initialize state tracking
        self.state = DesireSystemState(
            desires=np.zeros(len(profile_desires)),
            stability=np.ones(len(profile_desires)) * 0.5,  # Start with moderate stability
            coherence=np.ones(len(profile_desires)) * 0.5,  # Start with moderate coherence
            flow_rates=np.ones(len(profile_desires)) * 0.1  # Start with low flow rates
        )
    
    def calculate_desire_conflicts(self, desire_states: List[DesireState]) -> Dict[Tuple[int, int], float]:
        """
        Calculate conflicts between desires based on their categories and intensities.
        
        Args:
            desire_states: List of active desire states
            
        Returns:
            Dictionary mapping desire pairs to conflict scores
        """
        conflicts = {}
        n_desires = len(desire_states)
        
        for i in range(n_desires):
            for j in range(i + 1, n_desires):
                d1, d2 = desire_states[i], desire_states[j]
                
                # Calculate base conflict score
                conflict_score = 0.0
                
                # Category-based conflicts
                if d1.base_desire.category == d2.base_desire.category:
                    # Same category desires compete for attention
                    conflict_score += 0.3
                
                # Intensity-based conflicts
                intensity_diff = abs(d1.intensity - d2.intensity)
                conflict_score += 0.2 * (1 - intensity_diff)
                
                # Special handling for sexual vs non-sexual desires
                if (d1.base_desire.category == "Sexual & Intimate") != (d2.base_desire.category == "Sexual & Intimate"):
                    conflict_score += 0.4  # Higher conflict between sexual and non-sexual desires
                
                if conflict_score > self.conflict_threshold:
                    conflicts[(i, j)] = conflict_score
        
        return conflicts
    
    def resolve_conflicts(self, desire_states: List[DesireState], emotion_state: np.ndarray) -> List[DesireState]:
        """
        Resolve conflicts between desires based on emotions and traits.
        
        Args:
            desire_states: List of active desire states
            emotion_state: Current emotional state
            
        Returns:
            Updated list of desire states
        """
        conflicts = self.calculate_desire_conflicts(desire_states)
        
        for (i, j), conflict_score in conflicts.items():
            d1, d2 = desire_states[i], desire_states[j]
            
            # Calculate emotional influence on conflict resolution
            positive_emotion = emotion_state[0]
            negative_emotion = emotion_state[1]
            emotion_modulation = positive_emotion - negative_emotion
            
            # Resolve conflict based on emotions and base intensities
            if emotion_modulation > 0:
                # Positive emotions favor higher base intensity desires
                if d1.base_desire.importance * d1.base_desire.frequency > d2.base_desire.importance * d2.base_desire.frequency:
                    d2.intensity *= (1 - self.resolution_strength * conflict_score)
                else:
                    d1.intensity *= (1 - self.resolution_strength * conflict_score)
            else:
                # Negative emotions can suppress both desires
                d1.intensity *= (1 - self.resolution_strength * conflict_score * 0.5)
                d2.intensity *= (1 - self.resolution_strength * conflict_score * 0.5)
            
            # Ensure intensities don't fall below base values
            d1.intensity = max(d1.intensity, d1.base_desire.importance * d1.base_desire.frequency)
            d2.intensity = max(d2.intensity, d2.base_desire.importance * d2.base_desire.frequency)
        
        return desire_states
    
    def activate_desire(self, desire_state: DesireState, emotion_state: np.ndarray, trait_state: np.ndarray) -> None:
        """Activate a desire based on emotional and trait states."""
        # Calculate base intensity from importance and frequency
        base_intensity = desire_state.base_desire.importance * desire_state.base_desire.frequency
        
        # Calculate emotional influence (positive emotions increase, negative decrease)
        positive_emotion = emotion_state[0]  # First emotion is positive
        negative_emotion = emotion_state[1]  # Second emotion is negative
        emotion_modulation = positive_emotion - negative_emotion  # Positive increases, negative decreases
        
        # Calculate trait influence
        trait_influence = np.mean(np.abs(trait_state))
        
        # Set new intensity, allowing both increase and decrease but never below base
        # emotion_modulation ranges from -1 to 1, so we scale it appropriately
        intensity_modulation = 1.0 + 0.5 * emotion_modulation  # This gives range 0.5 to 1.5
        new_intensity = base_intensity * intensity_modulation
        desire_state.intensity = max(new_intensity, base_intensity)
        desire_state.is_active = True
        desire_state.age = 0.0
    
    def evolve(self, emotion_state: np.ndarray, trait_state: np.ndarray, dt: float) -> None:
        """Evolve the desire formation system."""
        # Update desire states
        for i, desire_state in enumerate(self.desires):
            # Calculate emotional influence
            emotional_influence = np.mean(emotion_state) * desire_state.base_desire.emotional_sensitivity
            
            # Calculate trait influence
            trait_influence = np.mean(trait_state) * desire_state.base_desire.trait_sensitivity
            
            # Update desire intensity
            base_intensity = desire_state.base_desire.importance * desire_state.base_desire.frequency
            desire_state.intensity = np.clip(
                desire_state.intensity + dt * (
                    emotional_influence +
                    trait_influence +
                    base_intensity * (1 - desire_state.intensity)
                ),
                0.0, 2.0
            )
            
            # Update desire age
            desire_state.age += dt
            
            # Update desire state
            self.state.desires[i] = desire_state.intensity
            
            # Update stability
            self.state.stability[i] = np.clip(
                self.state.stability[i] + dt * (
                    (1 - abs(desire_state.intensity - base_intensity)) * 0.1 -
                    self.state.stability[i] * 0.05
                ),
                0.1, 1.0
            )
            
            # Update coherence
            self.state.coherence[i] = np.clip(
                self.state.coherence[i] + dt * (
                    (1 - abs(desire_state.intensity - base_intensity)) * 0.1 -
                    self.state.coherence[i] * 0.05
                ),
                0.1, 1.0
            )
            
            # Update flow rates
            self.state.flow_rates[i] = np.clip(
                self.state.flow_rates[i] + dt * (
                    (1 - abs(desire_state.intensity - base_intensity)) * 0.1 -
                    self.state.flow_rates[i] * 0.05
                ),
                0.01, 0.5
            )
    
    def get_active_desires(self) -> List[DesireState]:
        """Get the list of active desires."""
        return [d for d in self.desires if d.is_active]
    
    def get_desire_categories(self) -> List[str]:
        """Get list of all desire categories from the profile."""
        return list(set(d.base_desire.category for d in self.desires))
    
    def get_desire_state(self) -> DesireSystemState:
        """Get the current state of the desire system."""
        return self.state 