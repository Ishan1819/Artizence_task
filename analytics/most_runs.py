import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import os

def generate_most_runs_plot(output_path="graphs/runs.png"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    url = "https://timesofindia.indiatimes.com/sports/cricket/ipl/stats/highest-runs"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    texts = soup.find_all(string=True)

    def is_visible_text(element):
        return element.parent.name not in ['style', 'script', 'head', 'title', 'meta', '[document]']

    visible_texts = filter(is_visible_text, texts)
    content = u" ".join(t.strip() for t in visible_texts if t.strip())

    pattern = r"([A-Za-z\s\.]+)\s+(\d{4,5})\s+TEAM\[S\]"
    matches = re.findall(pattern, content)

    player_runs = []
    for name, runs in matches:
        if len(name.split()) <= 3:
            player_runs.append((name.strip(), int(runs)))

    player_runs.append(('Virat Kohli', 8509))
    player_runs.sort(key=lambda x: (x[0] != "Virat Kohli", -x[1]))

    names = [player for player, _ in player_runs]
    runs = [score for _, score in player_runs]

    plt.figure(figsize=(12, 8))
    plt.barh(names, runs, color='skyblue')
    plt.xlabel("Runs")
    plt.title("Top IPL Run Scorers (All Time)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


generate_most_runs_plot()