import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from .personal_profile import PROFILE

def plot_system_evolution(
    trait_history: np.ndarray,
    emotion_history: np.ndarray,
    desire_history: np.ndarray,
    profile_metrics: Dict[str, np.ndarray],
    profile_system
) -> None:
    """Plot the evolution of the emotional-trait-desire system."""
    
    plt.figure(figsize=(20, 15))
    
    # Plot traits
    plt.subplot(4, 2, 1)
    for j in range(profile_system.n_traits):
        plt.plot(trait_history[:, j], label=f'Trait {j+1}')
    plt.title('Trait Evolution (Profile-Influenced)')
    plt.xlabel('Time')
    plt.ylabel('Trait Value')
    plt.legend()
    
    # Plot emotions
    plt.subplot(4, 2, 2)
    for j in range(profile_system.n_emotions):
        plt.plot(emotion_history[:, j], label=f'Emotion {j+1}')
    plt.title('Emotional Field Evolution (Profile-Influenced)')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.legend()
    
    # Plot desires with profile comparison
    plt.subplot(4, 2, 3)
    for j in range(profile_system.n_desires):
        plt.plot(desire_history[:, j], label=f'Desire {j+1}')
    # Add horizontal lines for profile desire values
    for j, desire in enumerate(PROFILE.desires):
        plt.axhline(y=desire.importance * desire.frequency, color='gray', linestyle='--', alpha=0.3)
    plt.title('Desire Evolution (Profile-Influenced)')
    plt.xlabel('Time')
    plt.ylabel('Desire Value')
    plt.legend()
    
    # Plot profile metrics
    plt.subplot(4, 2, 4)
    for key, values in profile_metrics.items():
        plt.plot(values, label=key.replace('_', ' ').title())
    plt.title('Profile Alignment Metrics')
    plt.xlabel('Time')
    plt.ylabel('Alignment')
    plt.legend()
    
    # Plot trait weights from profile
    plt.subplot(4, 2, 5)
    trait_weights = list(profile_system.trait_weights.values())
    plt.bar(range(len(trait_weights)), trait_weights)
    plt.title('Profile Trait Weights')
    plt.xlabel('Trait Index')
    plt.ylabel('Weight')
    
    # Plot emotion correlations from profile
    plt.subplot(4, 2, 6)
    emotion_correlations = [
        np.mean(list(correlations.values()))
        for correlations in profile_system.emotion_trait_correlations.values()
    ]
    plt.bar(range(len(emotion_correlations)), emotion_correlations)
    plt.title('Profile Emotion Correlations')
    plt.xlabel('Emotion Index')
    plt.ylabel('Correlation')
    
    # Plot desire importance from profile
    plt.subplot(4, 2, 7)
    desire_importance = [desire.importance for desire in PROFILE.desires]
    plt.bar(range(len(desire_importance)), desire_importance)
    plt.title('Profile Desire Importance')
    plt.xlabel('Desire Index')
    plt.ylabel('Importance')
    
    # Plot desire frequency from profile
    plt.subplot(4, 2, 8)
    desire_frequency = [desire.frequency for desire in PROFILE.desires]
    plt.bar(range(len(desire_frequency)), desire_frequency)
    plt.title('Profile Desire Frequency')
    plt.xlabel('Desire Index')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show() 