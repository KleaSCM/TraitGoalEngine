# test_goalEngine.py

from goalsEngine import GoalEngine


def test_goal_engine_basic():
    traits = {
        "Curiosity": {
            "value": 0.8,
            "goal_influence": {"Explore": 0.7, "Learn": 0.9}
        },
        "Anxiety": {
            "value": 0.3,
            "goal_influence": {"AvoidRisk": 1.0}
        },
        "Ambition": {
            "value": 0.5,
            "goal_influence": {"Achieve": 1.0, "Explore": 0.4}
        }
    }
    coupling = {
        'Anxiety': {'Curiosity': -0.3, 'Ambition': 0.1},
        'Curiosity': {'Anxiety': -0.2},
        'Ambition': {'Anxiety': -0.1}
    }

    engine = GoalEngine(traits=traits, coupling_matrix=coupling, decay_rate=0.05)
    engine.update_goals()

    # Assert goals created correctly
    assert "Explore" in engine.goals
    assert "Learn" in engine.goals
    assert "AvoidRisk" in engine.goals
    assert "Achieve" in engine.goals

    # Assert priorities sum to approx 1.0 (softmax)
    total_priority = sum(g.priority for g in engine.goals.values())
    assert abs(total_priority - 1.0) < 1e-6

    # Assert priorities are floats between 0 and 1
    for g in engine.goals.values():
        assert 0.0 <= g.priority <= 1.0

    # Progress test: advance 'Explore' goal by 0.2
    old_progress = engine.goals["Explore"].progress
    engine.progress_goal("Explore", 0.2)
    new_progress = engine.goals["Explore"].progress
    assert new_progress > old_progress
    assert new_progress <= 1.0

    # Test selecting top 2 goals returns 2 goals sorted by priority
    top_goals = engine.select_top_goals(2)
    assert len(top_goals) == 2
    assert top_goals[0].priority >= top_goals[1].priority

    print("All GoalEngine tests passed.")

if __name__ == "__main__":
    test_goal_engine_basic()
