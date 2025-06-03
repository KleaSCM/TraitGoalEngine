import numpy as np
from typing import Tuple, List, Optional
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
        
        # Initialize all profile desires as inactive
        for desire in profile_desires:
            self.desires.append(DesireState(
                base_desire=desire,
                intensity=0.0,
                decay_rate=0.01,
                is_active=False
            ))
    
    def activate_desire(self, desire_state: DesireState, emotion_state: np.ndarray, trait_state: np.ndarray) -> None:
        """Activate a desire based on emotional and trait states."""
        # Calculate activation probability based on desire's category and current states
        base_intensity = desire_state.base_desire.importance * desire_state.base_desire.frequency
        
        # Adjust intensity based on emotional and trait states
        emotion_influence = np.mean(np.abs(emotion_state))
        trait_influence = np.mean(np.abs(trait_state))
        
        # Set new intensity
        desire_state.intensity = base_intensity * (0.5 + 0.5 * (emotion_influence + trait_influence) / 2)
        desire_state.is_active = True
        desire_state.age = 0.0
    
    def evolve(self, emotion_state: np.ndarray, trait_state: np.ndarray, dt: float) -> np.ndarray:
        """Evolve the desire states based on emotional and trait influences."""
        # Update existing desires
        for desire_state in self.desires:
            if desire_state.is_active:
                # Decay the desire
                desire_state.intensity -= desire_state.decay_rate * dt
                desire_state.age += dt
                
                # Deactivate if intensity is too low
                if desire_state.intensity < 0.1:
                    desire_state.is_active = False
            else:
                # Try to activate inactive desires
                if np.random.random() < 0.01:  # Small chance to try activation each step
                    self.activate_desire(desire_state, emotion_state, trait_state)
        
        # Return current desire intensities
        return np.array([d.intensity for d in self.desires])
    
    def get_active_desires(self) -> List[DesireState]:
        """Get list of currently active desires."""
        return [d for d in self.desires if d.is_active]
    
    def get_desire_categories(self) -> List[str]:
        """Get list of all desire categories from the profile."""
        return list(set(d.base_desire.category for d in self.desires)) 