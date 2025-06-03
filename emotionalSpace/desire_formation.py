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
        self.conflict_threshold = 0.6  # Decreased from 0.7
        self.resolution_strength = 0.4  # Increased from 0.3
        self.emotional_modulation = 0.7  # Increased from 0.5
        self.trait_modulation = 0.7  # Added trait modulation parameter
        self.decay_rate = 0.15  # Added decay rate parameter
        
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
    
    def evolve(self, emotions: np.ndarray, traits: np.ndarray, dt: float) -> None:
        """Evolve the desire system with enhanced dynamics."""
        # Calculate conflicts with stronger threshold
        conflicts = self.calculate_desire_conflicts(self.desires)
        
        # Resolve conflicts with stronger resolution
        self.desires = self.resolve_conflicts(self.desires, emotions)
        
        # Calculate emotional and trait influences with stronger modulation
        emotional_influence = np.mean(emotions) * self.emotional_modulation * 0.8  # Reduced from 1.2
        trait_influence = np.mean(traits) * self.trait_modulation * 0.8  # Reduced from 1.2
        
        # Update desire intensities with stronger decay and lower maximum
        for desire in self.desires:
            # Calculate base evolution with reduced influence
            base_evolution = (
                emotional_influence * desire.base_desire.importance * 0.7 +  # Reduced from 1.0
                trait_influence * desire.base_desire.frequency * 0.7  # Reduced from 1.0
            )
            
            # Add random fluctuations
            random_factor = np.random.normal(0, 0.15)  # Reduced from 0.2
            
            # Update intensity with stronger decay and lower maximum
            desire.intensity = np.clip(
                desire.intensity + dt * (
                    base_evolution * (1 + random_factor) -
                    self.decay_rate * desire.intensity * 2.0  # Increased decay
                ),
                0.0, 1.0  # Reduced maximum from 1.2
            )
            
            # Update age
            desire.age += dt
        
        # Update system metrics with stronger decay
        self.state.stability = np.clip(
            self.state.stability * (1 - self.decay_rate * dt * 2.0) +  # Increased decay
            dt * np.mean([d.intensity for d in self.desires]) * 0.7,  # Reduced influence
            0.0, 0.7  # Reduced maximum from 0.8
        )
        
        self.state.coherence = np.clip(
            self.state.coherence * (1 - self.decay_rate * dt * 2.0) +  # Increased decay
            dt * (1 - np.mean(list(conflicts.values())) if conflicts else 0.0) * 0.7,  # Reduced influence
            0.0, 0.7  # Reduced maximum from 0.8
        )
        
        # Update flow rates with stronger decay
        for i, desire in enumerate(self.desires):
            self.state.flow_rates[i] = np.clip(
                self.state.flow_rates[i] * (1 - self.decay_rate * dt * 2.5) +  # Increased decay
                dt * desire.intensity * 0.7,  # Reduced influence
                0.0, 0.15  # Reduced maximum from 0.2
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