import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.predict_player_stats import ask_gemini 

@pytest.fixture
def mock_gemini_response():
    mock_response = MagicMock()
    mock_response.text = (
        "Based on the last 5 matches, Virat Kohli is expected to score 50+ runs with a strike rate of 135."
    )
    return mock_response


@patch("backend.models.predict_player_stats.model.generate_content")
def test_ask_gemini_returns_text(mock_generate_content, mock_gemini_response):
    mock_generate_content.return_value = mock_gemini_response

    player = "Virat Kohli"
    opponent = "Chennai Super Kings"
    response = ask_gemini(player, opponent)

    assert isinstance(response, str)
    assert player in response or "score" in response.lower()
    assert "runs" in response.lower() or "wickets" in response.lower()
