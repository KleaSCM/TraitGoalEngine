import pytest
from traitGoalEngine import calculate_goal_priorities, softmax

def test_goal_priority_computation():
    traits = {
        "Curiosity": {
            "value": 0.9,
            "goal_influence": {"Explore": 0.6, "Learn": 0.8}
        },
        "Fear": {
            "value": 0.4,
            "goal_influence": {"AvoidRisk": 1.0}
        },
        "Ambition": {
            "value": 0.7,
            "goal_influence": {"Achieve": 1.0}
        }
    }

    coupling = {
        "Curiosity": {"Fear": -0.2},
        "Fear": {"Curiosity": -0.1},
        "Ambition": {"Fear": 0.3}
    }

    decay = 0.1

    priorities = calculate_goal_priorities(traits, coupling_matrix=coupling, decay_rate=decay)

    assert isinstance(priorities, dict)
    assert set(priorities.keys()) == {"Explore", "Learn", "AvoidRisk", "Achieve"}

    total = sum(priorities.values())
    assert abs(total - 1.0) < 1e-6  # Softmax should normalize to 1

    # Optional: Check if higher Curiosity boosts Learn over AvoidRisk
    assert priorities["Learn"] > priorities["AvoidRisk"]

def test_zero_traits_still_returns_uniform():
    traits = {
        "Curiosity": {"value": 0.0, "goal_influence": {"Explore": 0.5}},
        "Ambition": {"value": 0.0, "goal_influence": {"Achieve": 0.5}}
    }

    result = calculate_goal_priorities(traits)

    assert set(result.keys()) == {"Explore", "Achieve"}
    assert abs(sum(result.values()) - 1.0) < 1e-6

def test_empty_goal_influence_trait():
    traits = {
        "Curiosity": {"value": 0.7, "goal_influence": {"Explore": 1.0}},
        "UnusedTrait": {"value": 0.9, "goal_influence": {}}
    }

    result = calculate_goal_priorities(traits)

    assert "Explore" in result
    assert isinstance(result["Explore"], float)
    assert abs(sum(result.values()) - 1.0) < 1e-6
