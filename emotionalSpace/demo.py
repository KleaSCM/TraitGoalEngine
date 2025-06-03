import numpy as np
import time
from emotionalSpace.system_integration import SystemIntegration
from emotionalSpace.profile_integration import ProfileIntegration
from emotionalSpace.personal_profile import PROFILE

def print_state(step: int, trait_state: np.ndarray, emotion_state: np.ndarray, desire_states, metrics: dict, profile_system):
    """Print the current state of the system."""
    print(f"\nStep {step}")
    print("=" * 80)
    
    # Print emotions
    print("\nEmotions:")
    print("-" * 40)
    emotions = ["positive", "negative"]
    for i, emotion in enumerate(emotions):
        print(f"{emotion:10}: {emotion_state[i]:.3f}")
    
    # Print traits
    print("\nTraits:")
    print("-" * 40)
    for i, (trait, weight) in enumerate(profile_system.trait_weights.items()):
        print(f"{trait:15}: {trait_state[i]:.3f}")
    
    # Print active desires
    print("\nActive Desires:")
    print("-" * 40)
    active_desires = [d for d in desire_states if d.is_active]
    for desire in sorted(active_desires, key=lambda x: x.intensity, reverse=True):
        base_intensity = desire.base_desire.importance * desire.base_desire.frequency
        print(f"{desire.base_desire.description:40} | Current: {desire.intensity:.3f} | Base: {base_intensity:.3f} | Age: {desire.age:.1f} | Category: {desire.base_desire.category}")
    
    # Print metrics
    print("\nMetrics:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key:20}: {value:.3f}")

def run_demo():
    """Run a demonstration of the emotional-trait-desire system with profile integration."""
    
    # Initialize system with profile integration
    profile_system = ProfileIntegration(PROFILE)
    
    # Simulation parameters
    dt = 0.01
    print_interval = 100  # Print every 100 steps
    step = 0
    
    try:
        while True:  # Run continuously until interrupted
            # Evolve system with profile influence
            profile_system.evolve_with_profile(dt)
            
            # Get current state with profile influence
            trait_state, emotion_state, desire_states = profile_system.get_profile_influenced_state()
            
            # Get profile metrics
            metrics = profile_system.get_profile_metrics()
            
            # Print state periodically
            if step % print_interval == 0:
                print_state(step, trait_state, emotion_state, desire_states, metrics, profile_system)
                time.sleep(0.1)  # Small delay for readability
            
            step += 1
            
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")
        print("\nFinal State:")
        print("=" * 80)
        print_state(step, trait_state, emotion_state, desire_states, metrics, profile_system)

if __name__ == '__main__':
    run_demo()
