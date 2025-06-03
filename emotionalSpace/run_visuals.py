import numpy as np
import matplotlib.pyplot as plt
from emotionalSpace.system_integration import SystemIntegration
from emotionalSpace.profile_integration import ProfileIntegration
from emotionalSpace.personal_profile import PROFILE
from emotionalSpace.visuals import plot_system_evolution, plot_phase_space

def run_continuous_visuals() -> None:
    """Run continuous visualizations of the emotional-trait-desire system."""
    # Initialize system
    profile_system = ProfileIntegration(PROFILE)
    
    # Simulation parameters
    dt = 0.1
    history_length = 100  # Keep last 100 steps for visualization
    
    # Initialize history arrays
    trait_history = np.zeros((history_length, len(profile_system.trait_weights)))
    emotion_history = np.zeros((history_length, 2))  # positive and negative emotions
    desire_history = np.zeros((history_length, profile_system.n_desires))
    
    # Initialize profile metrics history
    profile_metrics = {
        'trait_alignment': np.zeros(history_length),
        'emotion_alignment': np.zeros(history_length),
        'desire_alignment': np.zeros(history_length),
        'overall_alignment': np.zeros(history_length)
    }
    
    # Create figure for real-time plotting
    plt.ion()  # Enable interactive mode
    fig = plt.figure(figsize=(15, 12))
    
    step = 0
    try:
        while True:  # Run indefinitely
            # Get current state
            trait_state, emotion_state, desire_states = profile_system.get_profile_influenced_state()
            
            # Update history arrays (rolling window)
            trait_history = np.roll(trait_history, -1, axis=0)
            emotion_history = np.roll(emotion_history, -1, axis=0)
            desire_history = np.roll(desire_history, -1, axis=0)
            
            # Store current states
            trait_history[-1] = trait_state
            emotion_history[-1] = emotion_state
            
            # Store desire intensities
            for j, desire in enumerate(desire_states):
                if j < profile_system.n_desires:  # Ensure we don't exceed array bounds
                    desire_history[-1, j] = desire.intensity
            
            # Store profile metrics
            metrics = profile_system.get_profile_metrics()
            for key in profile_metrics:
                profile_metrics[key] = np.roll(profile_metrics[key], -1)
                profile_metrics[key][-1] = metrics[key]
            
            # Evolve system
            profile_system.evolve_with_profile(dt)
            
            # Update plots every 10 steps
            if step % 10 == 0:
                plt.clf()  # Clear current figure
                
                # Plot system evolution
                plot_system_evolution(
                    trait_history=trait_history,
                    emotion_history=emotion_history,
                    desire_history=desire_history,
                    profile_metrics=profile_metrics,
                    profile_system=profile_system
                )
                
                # Plot phase space
                plot_phase_space(
                    trait_history=trait_history,
                    emotion_history=emotion_history,
                    desire_history=desire_history,
                    profile_system=profile_system
                )
                
                plt.draw()
                plt.pause(0.001)  # Small pause to allow plot to update
            
            step += 1
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        plt.ioff()  # Disable interactive mode
        plt.show()  # Show final state

def run_single_visualization(profile_system: ProfileIntegration, n_steps: int = 100) -> None:
    """Run a single visualization of the system state."""
    # Simulation parameters
    dt = 0.1
    
    # Initialize history arrays
    trait_history = np.zeros((n_steps, len(profile_system.trait_weights)))
    emotion_history = np.zeros((n_steps, 2))  # positive and negative emotions
    desire_history = np.zeros((n_steps, profile_system.n_desires))
    
    # Initialize profile metrics history
    profile_metrics = {
        'trait_alignment': np.zeros(n_steps),
        'emotion_alignment': np.zeros(n_steps),
        'desire_alignment': np.zeros(n_steps),
        'overall_alignment': np.zeros(n_steps)
    }
    
    # Run simulation
    for i in range(n_steps):
        # Get current state
        trait_state, emotion_state, desire_states = profile_system.get_profile_influenced_state()
        
        # Store states
        trait_history[i] = trait_state
        emotion_history[i] = emotion_state
        
        # Store desire intensities
        for j, desire in enumerate(desire_states):
            if j < profile_system.n_desires:  # Ensure we don't exceed array bounds
                desire_history[i, j] = desire.intensity
        
        # Store profile metrics
        metrics = profile_system.get_profile_metrics()
        for key in profile_metrics:
            profile_metrics[key][i] = metrics[key]
        
        # Evolve system
        profile_system.evolve_with_profile(dt)
    
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Plot system evolution
    plot_system_evolution(
        trait_history=trait_history,
        emotion_history=emotion_history,
        desire_history=desire_history,
        profile_metrics=profile_metrics,
        profile_system=profile_system
    )
    
    # Plot phase space
    plot_phase_space(
        trait_history=trait_history,
        emotion_history=emotion_history,
        desire_history=desire_history,
        profile_system=profile_system
    )
    
    plt.show()

if __name__ == '__main__':
    run_continuous_visuals() 