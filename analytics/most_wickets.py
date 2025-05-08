# import requests
# from bs4 import BeautifulSoup
# import re
# import matplotlib.pyplot as plt

# def is_visible_text(element):
#     # Filter out non-visible elements
#     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
#         return False
#     return True

# # URL and headers to mimic a browser
# url = "https://timesofindia.indiatimes.com/sports/cricket/ipl/stats/most-wickets"
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# }

# # Request and parse the page
# response = requests.get(url, headers=headers)
# soup = BeautifulSoup(response.text, 'html.parser')
# text_elements = soup.find_all(string=True)
# visible_texts = filter(is_visible_text, text_elements)

# # Combine visible texts into one string
# content = u" ".join(t.strip() for t in visible_texts if t.strip())

# # Extract name and wickets using regex (TEAM[S] pattern is unique marker)
# pattern = r"([A-Za-z\s\.]+)\s+(\d{1,3})\s+TEAM\[S\]"
# matches = re.findall(pattern, content)

# # Filter valid player names (1 to 3 words max)
# player_wickets = []
# for name, wickets in matches:
#     if len(name.split()) <= 3:  # Likely a player name
#         player_wickets.append((name.strip(), int(wickets)))

# # Manually add a popular bowler's wickets (example: Lasith Malinga with 170 wickets)
# player_wickets.append(('Yuzvendra Chahal', 219))

# # Sort by wickets (Lasith Malinga always first if present)
# player_wickets.sort(key=lambda x: (x[0] != "Yuzvendra Chahal", -x[1]))

# # Split names and wickets for plotting
# names = [player for player, _ in player_wickets]
# wickets = [score for _, score in player_wickets]

# # Plot horizontal bar chart
# plt.figure(figsize=(12, 8))
# plt.barh(names, wickets, color='lightgreen')
# plt.xlabel("Wickets")
# plt.title("Top IPL Wicket-Takers (All Time)")
# plt.gca().invert_yaxis()  # Highest wicket-taker on top
# plt.tight_layout()
# plt.show()


import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt

def generate_most_wickets_plot(output_path="graphs/most_wickets.png"):
    url = "https://timesofindia.indiatimes.com/sports/cricket/ipl/stats/most-wickets"
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

    pattern = r"([A-Za-z\s\.]+)\s+(\d{1,3})\s+TEAM\[S\]"
    matches = re.findall(pattern, content)

    player_wickets = []
    for name, wickets in matches:
        if len(name.split()) <= 3:
            player_wickets.append((name.strip(), int(wickets)))

    player_wickets.append(('Yuzvendra Chahal', 219))
    player_wickets.sort(key=lambda x: (x[0] != "Yuzvendra Chahal", -x[1]))

    names = [player for player, _ in player_wickets]
    wickets = [score for _, score in player_wickets]

    plt.figure(figsize=(12, 8))
    plt.barh(names, wickets, color='lightgreen')
    plt.xlabel("Wickets")
    plt.title("Top IPL Wicket-Takers (All Time)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


generate_most_wickets_plot()