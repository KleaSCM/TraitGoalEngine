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
    
    # Plot traits with coupling visualization
    plt.subplot(4, 2, 1)
    for j in range(profile_system.n_traits):
        plt.plot(trait_history[:, j], label=f'Trait {j+1}')
    plt.title('Trait Evolution (Profile-Influenced)')
    plt.xlabel('Time')
    plt.ylabel('Trait Value')
    plt.legend()
    
    # Plot trait interaction matrix
    plt.subplot(4, 2, 2)
    interaction_matrix = profile_system.trait_evolution.state.interaction_matrix
    plt.imshow(interaction_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Interaction Strength')
    plt.title('Trait Interaction Matrix')
    plt.xlabel('Trait Index')
    plt.ylabel('Trait Index')
    
    # Plot emotions
    plt.subplot(4, 2, 3)
    for j in range(profile_system.n_emotions):
        plt.plot(emotion_history[:, j], label=f'Emotion {j+1}')
    plt.title('Emotional Field Evolution (Profile-Influenced)')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.legend()
    
    # Plot trait-emotion coupling
    plt.subplot(4, 2, 4)
    coupling_matrix = profile_system.trait_evolution.state.coupling_strength[:, :profile_system.n_emotions]
    plt.imshow(coupling_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Coupling Strength')
    plt.title('Trait-Emotion Coupling Matrix')
    plt.xlabel('Emotion Index')
    plt.ylabel('Trait Index')
    
    # Plot desires with profile comparison
    plt.subplot(4, 2, 5)
    for j in range(profile_system.n_desires):
        plt.plot(desire_history[:, j], label=f'Desire {j+1}')
    # Add horizontal lines for profile desire values
    for j, desire in enumerate(PROFILE.desires):
        plt.axhline(y=desire.importance * desire.frequency, color='gray', linestyle='--', alpha=0.3)
    plt.title('Desire Evolution (Profile-Influenced)')
    plt.xlabel('Time')
    plt.ylabel('Desire Value')
    plt.legend()
    
    # Plot trait-desire coupling
    plt.subplot(4, 2, 6)
    desire_coupling = profile_system.trait_evolution.state.coupling_strength[:, profile_system.n_emotions:]
    plt.imshow(desire_coupling, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Coupling Strength')
    plt.title('Trait-Desire Coupling Matrix')
    plt.xlabel('Desire Index')
    plt.ylabel('Trait Index')
    
    # Plot trait stability
    plt.subplot(4, 2, 7)
    stability = profile_system.trait_evolution.state.stability
    plt.bar(range(len(stability)), stability)
    plt.title('Trait Stability')
    plt.xlabel('Trait Index')
    plt.ylabel('Stability')
    
    # Plot trait learning rates
    plt.subplot(4, 2, 8)
    learning_rates = profile_system.trait_evolution.state.learning_rate
    plt.bar(range(len(learning_rates)), learning_rates)
    plt.title('Trait Learning Rates')
    plt.xlabel('Trait Index')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.show() 