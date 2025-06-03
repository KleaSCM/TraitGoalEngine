import numpy as np
import time
from emotionalSpace.system_integration import SystemIntegration
from emotionalSpace.profile_integration import ProfileIntegration
from emotionalSpace.personal_profile import PROFILE

def print_state(profile_system: ProfileIntegration, step: int) -> None:
    """Print the current state of the system."""
    print(f"\nStep {step}")
    print("=" * 80)
    
    # Get current states
    trait_state, emotion_state, desire_states = profile_system.get_profile_influenced_state()
    trait_names = list(profile_system.trait_weights.keys())
    
    # Print emotions
    print("\nEmotions:")
    print("-" * 40)
    for i, emotion in enumerate(["positive", "negative"]):
        print(f"{emotion:<10}: {emotion_state[i]:.3f}")
    
    # Print traits with metrics
    print("\nTraits:")
    print("-" * 40)
    for i, (name, value) in enumerate(zip(trait_names, trait_state)):
        print(f"{name:<12}: {value:.3f}")
        print(f"  Stability: {profile_system.trait_evolution.state.stability[i]:.3f}")
        print(f"  Learning Rate: {profile_system.trait_evolution.state.learning_rate[i]:.3f}")
        print(f"  Interaction Strength: {np.mean(np.abs(profile_system.trait_evolution.state.interaction_matrix[i])):.3f}")
        print(f"  Emotion Coupling: {np.mean(np.abs(profile_system.trait_evolution.state.coupling_strength[i, :2])):.3f}")
        print(f"  Desire Coupling: {np.mean(np.abs(profile_system.trait_evolution.state.coupling_strength[i, 2:])):.3f}")
    
    # Print active desires
    print("\nActive Desires:")
    print("-" * 40)
    for desire in desire_states:
        print(f"{desire.base_desire.description:<40} | Current: {desire.intensity:.3f} | Base: {desire.base_desire.importance * desire.base_desire.frequency:.3f} | Age: {desire.age:.1f} | Category: {desire.base_desire.category}")
    
    # Print system metrics
    metrics = profile_system.get_system_metrics()
    print("\nSystem Metrics:")
    print("-" * 40)
    print(f"Stability: {metrics['stability']:.3f}")
    print(f"Performance: {metrics['performance']:.3f}")
    print(f"System Norm: {metrics['system_norm']:.3f}")
    
    # Print phase space metrics
    print("\nPhase Space Metrics:")
    print("-" * 40)
    print(f"Trait Volume: {metrics['trait_volume']:.3f}")
    print(f"Emotion Volume: {metrics['emotion_volume']:.3f}")
    print(f"Desire Volume: {metrics['desire_volume']:.3f}")
    print(f"Trait Attractor: {metrics['trait_attractor']:.3f}")
    print(f"Emotion Attractor: {metrics['emotion_attractor']:.3f}")
    print(f"Desire Attractor: {metrics['desire_attractor']:.3f}")
    
    # Print profile metrics
    profile_metrics = profile_system.get_profile_metrics()
    print("\nProfile Metrics:")
    print("-" * 40)
    for key, value in profile_metrics.items():
        print(f"{key:<20}: {value:.3f}")
    
    # Print trait interaction summary LAST
    print("\nTrait Interaction Summary:")
    print("-" * 40)
    interaction_matrix = profile_system.trait_evolution.state.interaction_matrix
    n_traits = len(trait_names)
    for i in range(n_traits):
        for j in range(n_traits):
            if i != j and abs(interaction_matrix[i, j]) > 0.1:
                print(f"Trait {i+1} → Trait {j+1}: {interaction_matrix[i, j]:.3f}")

def run_demo() -> None:
    """Run a demonstration of the emotional-trait-desire system."""
    # Initialize the system
    profile_system = ProfileIntegration(PROFILE)
    
    # Run simulation
    dt = 0.1
    step = 0
    
    try:
        while True:  # Run indefinitely
            # Print state every 10 steps
            if step % 10 == 0:
                print_state(profile_system, step)
            
            # Evolve system
            profile_system.evolve_with_profile(dt)
            step += 1
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
        # Print final state
        print_state(profile_system, step)

if __name__ == '__main__':
    run_demo()
