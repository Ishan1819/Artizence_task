import pytest
from unittest.mock import patch
from io import StringIO
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analytics')))

from analytics.winner_acc_years import plot_yearly_wins

@pytest.fixture
def mock_csv_data():
    data = """year,winner
2018,Team A
2018,Team B
2018,Team A
2019,Team A
2019,Team C
2019,Team B
2020,Team C
2020,Team B
2020,Team C
"""
    return StringIO(data)  

@patch('matplotlib.pyplot.savefig')  
def test_plot_yearly_wins(mock_savefig, mock_csv_data):
    # Given
    input_csv = mock_csv_data 
    
    plot_yearly_wins(input_csv, output_path='test_output.png')
    mock_savefig.assert_called_once_with('test_output.png')
