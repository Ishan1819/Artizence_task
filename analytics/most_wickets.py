import pytest
import os
import sys
import matplotlib
# Set the backend to 'Agg' to avoid the Tkinter issue
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch
from io import BytesIO

# Add the directory containing your 'analytics' module to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming generate_most_wickets_plot is imported from your script
from analytics.most_wickets import generate_most_wickets_plot

# Mocked response content for the test
mocked_html_content = """
<html>
    <body>
        <div class="stats">
            <span>Yuzvendra Chahal 219 TEAM[S]</span>
            <span>Lasith Malinga 170 TEAM[S]</span>
            <span>Ravichandran Ashwin 150 TEAM[S]</span>
        </div>
    </body>
</html>
"""

@pytest.fixture
def mock_request_get():
    with patch('requests.get') as mock_get:
        mock_get.return_value.text = mocked_html_content
        yield mock_get

def test_generate_most_wickets_plot(mock_request_get):
    output_path = "tests/test_graphs/wickets_test.png"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Call the function to generate the plot
    generate_most_wickets_plot(output_path)

    # Check if the plot is created
    assert os.path.exists(output_path), f"Plot was not created at {output_path}"

    # Check if the generated file is a valid image
    with open(output_path, "rb") as f:
        image_data = f.read()
        assert image_data[:4] == b'\x89PNG', "Generated file is not a valid PNG image"

    # Cleanup after test
    os.remove(output_path)
