import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.predict_winner_score import predict_match_score 
VALID_TEAM1 = "Mumbai Indians"
VALID_TEAM2 = "Chennai Super Kings"

INVALID_TEAM = "Unknown United"

def test_predict_match_score_output():
    result = predict_match_score(VALID_TEAM1, VALID_TEAM2)
    

    expected_keys = {
        "team1", "team2", "predicted_winner", "win_probability",
        "estimated_winner_score", "predicted_loser_score", "estimated_margin"
    }
    assert expected_keys.issubset(result.keys())

    # Type and value checks
    assert isinstance(result["team1"], str)
    assert isinstance(result["team2"], str)
    assert isinstance(result["predicted_winner"], str)
    assert isinstance(result["win_probability"], float)
    assert 0 <= result["win_probability"] <= 100
    assert isinstance(result["estimated_winner_score"], float)
    assert isinstance(result["predicted_loser_score"], int)
    assert isinstance(result["estimated_margin"], float)

def test_predict_with_unknown_team():
    with pytest.raises(ValueError, match="One of the teams not in training data"):
        predict_match_score(INVALID_TEAM, VALID_TEAM2)
