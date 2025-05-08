import pandas as pd
import matplotlib.pyplot as plt

def plot_yearly_wins(input_csv, output_path='yearly_wins.png'):
    """
    Function to plot year-wise wins by IPL teams in a stacked bar chart.
    
    Parameters:
    - input_csv (str): Path to the input CSV file containing IPL match data.
    - output_path (str): Path where the plot will be saved. Default is 'yearly_wins.png'.
    """
    # Load your dataset
    df = pd.read_csv(input_csv)

    # Group by year and winner, then count wins
    yearly_wins = df.groupby(['year', 'winner']).size().reset_index(name='wins')

    # Pivot the data to get winners as columns
    pivot_df = yearly_wins.pivot(index='year', columns='winner', values='wins').fillna(0)

    # Plot the stacked bar chart
    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind='bar', stacked=True, colormap='tab20')
    
    # Customize the plot
    plt.title('Year-wise Wins by IPL Teams')
    plt.xlabel('Year')
    plt.ylabel('Number of Wins')
    plt.legend(title='Teams', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)

    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()

# Example usage:
input_csv = 'D:/Ishan_ip datasets/cleaned_unique_matches.csv'  # Replace with your actual CSV file path
output_path = 'graphs/yearly_wins.png'  # Specify the desired output path
plot_yearly_wins(input_csv, output_path)
