import pytest
import os
import sys
import matplotlib.pyplot as plt
from unittest.mock import patch
from io import BytesIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analytics.most_runs import generate_most_runs_plot

mocked_html_content = """
<html>
    <body>
        <div class="stats">
            <span>Virat Kohli 8509 TEAM[S]</span>
            <span>Rohit Sharma 5400 TEAM[S]</span>
            <span>Chris Gayle 4900 TEAM[S]</span>
        </div>
    </body>
</html>
"""

@pytest.fixture
def mock_request_get():
    with patch('requests.get') as mock_get:
        mock_get.return_value.text = mocked_html_content
        yield mock_get

def test_generate_most_runs_plot(mock_request_get):
    output_path = "tests/test_graphs/runs_test.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    generate_most_runs_plot(output_path)

    assert os.path.exists(output_path), f"Plot was not created at {output_path}"

    with open(output_path, "rb") as f:
        image_data = f.read()
        assert image_data[:4] == b'\x89PNG', "Generated file is not a valid PNG image"

    os.remove(output_path)
