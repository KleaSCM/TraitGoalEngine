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
    
    def evolve(self, emotion_state: np.ndarray, trait_state: np.ndarray, dt: float) -> np.ndarray:
        """Evolve the desire states based on emotional and trait influences."""
        # Calculate emotional influences with enhanced feedback
        positive_emotion = emotion_state[0]
        negative_emotion = emotion_state[1]
        emotion_modulation = positive_emotion - negative_emotion
        
        # Get active desires
        active_desires = [d for d in self.desires if d.is_active]
        
        # Resolve conflicts between active desires
        active_desires = self.resolve_conflicts(active_desires, emotion_state)
        
        # Update existing desires with enhanced feedback
        for desire_state in self.desires:
            if desire_state.is_active:
                # Calculate desire-specific influences
                # Use default weights if not available
                if not hasattr(desire_state.base_desire, 'emotion_weights'):
                    desire_state.base_desire.emotion_weights = np.array([0.5, 0.5])
                if not hasattr(desire_state.base_desire, 'trait_weights'):
                    desire_state.base_desire.trait_weights = np.ones(len(trait_state)) / len(trait_state)
                
                # Calculate influences with stronger feedback
                desire_emotion_influence = np.dot(emotion_state, desire_state.base_desire.emotion_weights)
                desire_trait_influence = np.dot(trait_state, desire_state.base_desire.trait_weights)
                
                # Calculate base intensity
                base_intensity = desire_state.base_desire.importance * desire_state.base_desire.frequency
                
                # Calculate intensity modulation based on emotions with stronger effect
                # Positive emotions increase, negative emotions decrease
                intensity_modulation = 1.0 + emotion_modulation  # Increased from 0.5 to 1.0
                
                # Add random fluctuation to prevent stagnation
                random_fluctuation = np.random.normal(0, 0.05)
                
                # Calculate new intensity with stronger feedback and random fluctuation
                new_intensity = base_intensity * intensity_modulation * (1 + desire_trait_influence) + random_fluctuation
                
                # Ensure we never go below base intensity
                desire_state.intensity = max(new_intensity, base_intensity)
                
                desire_state.age += dt
                
                # Deactivate if age is too high, but maintain base intensity
                if desire_state.age > 1000:
                    desire_state.is_active = False
                    desire_state.intensity = base_intensity
            else:
                # Try to activate inactive desires with enhanced probability
                # Higher chance when positive emotions are stronger
                activation_prob = 0.1 * desire_state.base_desire.frequency * (1 + emotion_modulation)  # Increased from 0.05 to 0.1
                
                # Add trait influence to activation probability
                trait_activation = np.mean(np.abs(trait_state)) * 0.1
                activation_prob += trait_activation
                
                if np.random.random() < activation_prob:
                    self.activate_desire(desire_state, emotion_state, trait_state)
        
        # Return current desire intensities
        return np.array([d.intensity for d in self.desires])
    
    def get_active_desires(self) -> List[DesireState]:
        """Get list of currently active desires."""
        return [d for d in self.desires if d.is_active]
    
    def get_desire_categories(self) -> List[str]:
        """Get list of all desire categories from the profile."""
        return list(set(d.base_desire.category for d in self.desires)) 