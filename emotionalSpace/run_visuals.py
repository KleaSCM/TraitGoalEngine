import numpy as np
from emotionalSpace.system_integration import SystemIntegration
from emotionalSpace.profile_integration import ProfileIntegration
from emotionalSpace.personal_profile import PROFILE
from emotionalSpace.visuals import plot_system_evolution

def run_visuals():
    """Run the visualization of the emotional-trait-desire system."""
    
    # Initialize system with profile integration
    profile_system = ProfileIntegration(PROFILE)
    
    # Simulation parameters
    n_steps = 1000
    dt = 0.01
    
    # Storage for plotting
    trait_history = np.zeros((n_steps, profile_system.n_traits))
    emotion_history = np.zeros((n_steps, profile_system.n_emotions))
    desire_history = np.zeros((n_steps, profile_system.n_desires))
    profile_metrics = {
        'trait_alignment': np.zeros(n_steps),
        'emotion_alignment': np.zeros(n_steps),
        'desire_alignment': np.zeros(n_steps),
        'overall_alignment': np.zeros(n_steps)
    }
    
    # Run simulation
    for i in range(n_steps):
        # Evolve system with profile influence
        profile_system.evolve_with_profile(dt)
        
        # Get current state with profile influence
        trait_state, emotion_state, desire_state = profile_system.get_profile_influenced_state()
        
        # Store state for plotting
        trait_history[i] = trait_state
        emotion_history[i] = emotion_state
        desire_history[i] = desire_state
        
        # Store profile metrics
        metrics = profile_system.get_profile_metrics()
        for key in profile_metrics:
            profile_metrics[key][i] = metrics[key]
    
    # Plot the evolution
    plot_system_evolution(trait_history, emotion_history, desire_history, profile_metrics, profile_system)

if __name__ == '__main__':
    run_visuals() 