from typing import Dict, Any, List, Tuple
import math
from traitDict import traits, personal_traits


def sigmoid(x: float) -> float:
    """Smooth nonlinearity for trait value saturation between 0 and 1."""
    return 1 / (1 + math.exp(-x))


def softmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Convert unbounded goal scores to a normalized probability distribution."""
    if not scores:
        return {}
    max_score = max(scores.values())
    exps = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


def update_trait_values(traits_dict: Dict[str, Dict[str, Any]], 
                       goal_progress: Dict[str, float],
                       learning_rate: float = 0.1) -> Dict[str, float]:
    """
    Update trait values based on goal progress and trait interactions.
    Implements a continuous learning and adaptation mechanism.
    """
    updated_values = {}
    
    for trait_name, trait_data in traits_dict.items():
        # Get current base value
        current_value = trait_data.get('value', 0.0)
        
        # Calculate influence from goal progress
        goal_influence = 0.0
        for goal, progress in goal_progress.items():
            if goal in trait_data.get('goal_influence', {}):
                influence = trait_data['goal_influence'][goal]
                goal_influence += influence * progress
        
        # Calculate coupling influence
        coupling_influence = 0.0
        for linked_trait in trait_data.get('links', []):
            if linked_trait in traits_dict:
                linked_value = traits_dict[linked_trait].get('value', 0.0)
                coupling_influence += 0.3 * linked_value  # Coupling coefficient
        
        # Update value with learning and coupling
        new_value = current_value + learning_rate * (
            goal_influence +  # Goal progress influence
            coupling_influence -  # Trait coupling
            current_value * 0.05  # Natural decay
        )
        
        # Ensure value stays in [0,1]
        updated_values[trait_name] = max(0.0, min(1.0, new_value))
    
    return updated_values


def process_trait_values(traits_dict: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Process raw trait values into normalized values between 0 and 1.
    Uses valence and stability to compute the effective value.
    """
    processed = {}
    for name, data in traits_dict.items():
        if "valence" in data and "stability" in data:
            # Combine valence and stability into a single value
            base_value = (data["valence"] + 1) / 2  # Convert [-1,1] to [0,1]
            stability_factor = data["stability"]
            processed[name] = base_value * stability_factor
    return processed


def create_goal_influences(traits_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Create goal influence mappings from trait links and weights.
    """
    influences = {}
    for name, data in traits_dict.items():
        if "links" in data and "weight" in data:
            # Each trait influences goals based on its linked traits
            influences[name] = {
                "value": data.get("weight", 0.0),
                "goal_influence": {link: data["weight"] * 0.5 for link in data["links"]}
            }
    return influences


def apply_decay_and_sigmoid(trait_vals: Dict[str, float], decay: float = 0.0) -> Dict[str, float]:
    """
    Apply decay and sigmoid activation to processed trait values.
    Decay simulates temporal fatigue or attentional saturation.
    """
    adjusted = {}
    for name, value in trait_vals.items():
        decayed = value * (1 - decay)
        # Stretch sigmoid over a meaningful range centered at 0.5
        adjusted[name] = sigmoid(decayed * 5 - 2.5)
    return adjusted


def apply_trait_coupling(trait_vals: Dict[str, float], coupling: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Apply a symmetric or asymmetric inter-trait influence matrix.
    Each trait is adjusted based on others using defined coupling weights.
    """
    coupled = {}
    for t1 in trait_vals:
        influence_sum = trait_vals[t1]
        for t2, coeff in coupling.get(t1, {}).items():
            if t2 in trait_vals:
                influence_sum += coeff * trait_vals[t2]
        # Reapply sigmoid to resaturate
        coupled[t1] = sigmoid(influence_sum)
    return coupled


def aggregate_goal_scores(traits: Dict[str, Dict[str, Any]],
                          adjusted_values: Dict[str, float],
                          exponent: float = 1.5) -> Dict[str, float]:
    """
    Compute goal scores by summing over trait influences weighted by nonlinearly scaled trait values.
    """
    scores = {}
    for t_name, t_data in traits.items():
        t_weight = adjusted_values.get(t_name, 0.0)
        influences = t_data.get("goal_influence", {})
        for goal, weight in influences.items():
            contribution = weight * (t_weight ** exponent)
            scores[goal] = scores.get(goal, 0.0) + contribution
    return scores


def prepare_traits_for_goals_engine() -> Dict[str, Dict[str, Any]]:
    """
    Main function to prepare traits for the goals engine.
    Processes traits from traitDict.py and returns them in the format expected by goalsEngine.py
    """
    # Process trait values
    processed_values = process_trait_values(traits)
    
    # Create goal influences
    trait_influences = create_goal_influences(traits)
    
    # Combine into the format expected by goalsEngine
    engine_traits = {}
    for name in processed_values:
        if name in trait_influences:
            engine_traits[name] = {
                "value": processed_values[name],
                "goal_influence": trait_influences[name]["goal_influence"]
            }
    
    return engine_traits


def calculate_goal_priorities(decay_rate: float = 0.0) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Full trait → coupling → goal priority pipeline with continuous updates.
    
    Args:
        decay_rate: Optional decay/fatigue modifier.
    
    Returns:
        Tuple of (normalized goal priorities, updated traits)
    """
    # Get processed traits
    engine_traits = prepare_traits_for_goals_engine()
    
    # Extract just the values for decay and sigmoid processing
    trait_values = {name: data["value"] for name, data in engine_traits.items()}
    
    # Step 1: Adjust trait values (sigmoid+decay)
    adjusted_values = apply_decay_and_sigmoid(trait_values, decay=decay_rate)
    
    # Step 2: Create coupling matrix from trait links
    coupling_matrix = {}
    for name, data in traits.items():
        if "links" in data:
            coupling_matrix[name] = {link: 0.3 for link in data["links"]}
    
    # Apply trait coupling
    if coupling_matrix:
        adjusted_values = apply_trait_coupling(adjusted_values, coupling_matrix)
    
    # Step 3: Aggregate nonlinear trait contributions to goals
    goal_scores = aggregate_goal_scores(engine_traits, adjusted_values)
    
    # Step 4: Normalize into a probability distribution
    priorities = softmax(goal_scores)
    
    # Step 5: Update trait values based on goal progress
    updated_traits = update_trait_values(engine_traits, priorities)
    
    # Update the engine traits with new values
    for name, value in updated_traits.items():
        if name in engine_traits:
            engine_traits[name]["value"] = value
    
    return priorities, engine_traits
