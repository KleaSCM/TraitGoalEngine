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
    plt.figure(figsize=(15, 12))
    
    # Plot trait evolution
    plt.subplot(4, 2, 1)
    for i in range(trait_history.shape[1]):
        plt.plot(trait_history[:, i], label=f'Trait {i+1}')
    plt.title('Trait Evolution')
    plt.xlabel('Time')
    plt.ylabel('Trait Value')
    plt.legend()
    
    # Plot emotion evolution
    plt.subplot(4, 2, 2)
    plt.plot(emotion_history[:, 0], label='Positive')
    plt.plot(emotion_history[:, 1], label='Negative')
    plt.title('Emotion Evolution')
    plt.xlabel('Time')
    plt.ylabel('Emotion Intensity')
    plt.legend()
    
    # Plot desire evolution
    plt.subplot(4, 2, 3)
    for i in range(desire_history.shape[1]):
        plt.plot(desire_history[:, i], label=f'Desire {i+1}')
    plt.title('Desire Evolution')
    plt.xlabel('Time')
    plt.ylabel('Desire Intensity')
    plt.legend()
    
    # Plot profile metrics
    plt.subplot(4, 2, 4)
    for metric, values in profile_metrics.items():
        plt.plot(values, label=metric)
    plt.title('Profile Metrics')
    plt.xlabel('Time')
    plt.ylabel('Metric Value')
    plt.legend()
    
    # Plot trait interaction matrix
    plt.subplot(4, 2, 5)
    interaction_matrix = profile_system.trait_evolution.state.interaction_matrix
    plt.imshow(interaction_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Interaction Strength')
    plt.title('Trait Interaction Matrix')
    plt.xlabel('Trait Index')
    plt.ylabel('Trait Index')
    
    # Plot trait-emotion coupling
    plt.subplot(4, 2, 6)
    emotion_coupling = profile_system.trait_evolution.state.coupling_strength[:, :profile_system.n_emotions]
    plt.imshow(emotion_coupling, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Coupling Strength')
    plt.title('Trait-Emotion Coupling Matrix')
    plt.xlabel('Emotion Index')
    plt.ylabel('Trait Index')
    
    # Plot trait-desire coupling
    plt.subplot(4, 2, 7)
    desire_coupling = profile_system.trait_evolution.state.coupling_strength[:, profile_system.n_emotions:]
    plt.imshow(desire_coupling, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Coupling Strength')
    plt.title('Trait-Desire Coupling Matrix')
    plt.xlabel('Desire Index')
    plt.ylabel('Trait Index')
    
    # Plot phase space metrics
    plt.subplot(4, 2, 8)
    metrics = profile_system.get_system_metrics()
    phase_metrics = {
        'Trait Volume': metrics.get('trait_volume', 0),
        'Emotion Volume': metrics.get('emotion_volume', 0),
        'Desire Volume': metrics.get('desire_volume', 0),
        'Trait Attractor': metrics.get('trait_attractor', 0),
        'Emotion Attractor': metrics.get('emotion_attractor', 0),
        'Desire Attractor': metrics.get('desire_attractor', 0)
    }
    plt.bar(range(len(phase_metrics)), list(phase_metrics.values()))
    plt.xticks(range(len(phase_metrics)), list(phase_metrics.keys()), rotation=45)
    plt.title('Phase Space Metrics')
    plt.tight_layout()
    
    plt.show()

def plot_phase_space(
    trait_history: np.ndarray,
    emotion_history: np.ndarray,
    desire_history: np.ndarray,
    profile_system
) -> None:
    """Plot phase space analysis of the system."""
    plt.figure(figsize=(15, 10))
    
    # Plot trait phase space
    plt.subplot(2, 2, 1)
    for i in range(trait_history.shape[1]):
        for j in range(i+1, trait_history.shape[1]):
            plt.plot(trait_history[:, i], trait_history[:, j], 
                    label=f'Trait {i+1} vs {j+1}', alpha=0.5)
    plt.title('Trait Phase Space')
    plt.xlabel('Trait Value')
    plt.ylabel('Trait Value')
    plt.legend()
    
    # Plot emotion phase space
    plt.subplot(2, 2, 2)
    plt.plot(emotion_history[:, 0], emotion_history[:, 1], 'b-', alpha=0.5)
    plt.scatter(emotion_history[0, 0], emotion_history[0, 1], c='g', label='Start')
    plt.scatter(emotion_history[-1, 0], emotion_history[-1, 1], c='r', label='End')
    plt.title('Emotion Phase Space')
    plt.xlabel('Positive Emotion')
    plt.ylabel('Negative Emotion')
    plt.legend()
    
    # Plot desire phase space
    plt.subplot(2, 2, 3)
    for i in range(desire_history.shape[1]):
        for j in range(i+1, desire_history.shape[1]):
            plt.plot(desire_history[:, i], desire_history[:, j],
                    label=f'Desire {i+1} vs {j+1}', alpha=0.5)
    plt.title('Desire Phase Space')
    plt.xlabel('Desire Value')
    plt.ylabel('Desire Value')
    plt.legend()
    
    # Plot stability basins
    plt.subplot(2, 2, 4)
    metrics = profile_system.get_system_metrics()
    stability = metrics.get('stability', 0)
    performance = metrics.get('performance', 0)
    plt.scatter(stability, performance, c='b', s=100)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.title('Stability-Performance Basin')
    plt.xlabel('Stability')
    plt.ylabel('Performance')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show() 