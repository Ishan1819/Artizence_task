# fallback.py

import pandas as pd
from datetime import datetime
from config import DATASET_PATH

def fallback_response(question):
    ipl_teams = {
        "MI": "Mumbai Indians", "CSK": "Chennai Super Kings", "RCB": "Royal Challengers Bangalore",
        "KKR": "Kolkata Knight Riders", "DC": "Delhi Capitals", "SRH": "Sunrisers Hyderabad",
        "PBKS": "Punjab Kings", "RR": "Rajasthan Royals", "GT": "Gujarat Titans", "LSG": "Lucknow Super Giants"
    }
    for full in list(ipl_teams.values()):
        ipl_teams[full] = full

    mentioned = [name for abbr, name in ipl_teams.items() if abbr in question]
    if len(mentioned) < 2:
        return "• Please mention at least two IPL teams for analysis."

    team1, team2 = mentioned[0], mentioned[1]
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df = df[df['year'] >= datetime.now().year - 1]

    t1_matches = df[(df['team1'] == team1) | (df['team2'] == team1)].tail(5)
    t1_wins = len(t1_matches[t1_matches['winner'] == team1])

    t2_matches = df[(df['team1'] == team2) | (df['team2'] == team2)].tail(5)
    t2_wins = len(t2_matches[t2_matches['winner'] == team2])

    h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) |
             ((df['team1'] == team2) & (df['team2'] == team1))].tail(3)
    t1_h2h = len(h2h[h2h['winner'] == team1])
    t2_h2h = len(h2h[h2h['winner'] == team2])

    response = f"• {team1} won {t1_wins}/5 recent games, {team2} won {t2_wins}/5.\n"
    response += f"• Head-to-head: {team1} ({t1_h2h}), {team2} ({t2_h2h}).\n"
    response += f"• {team1 if t1_wins > t2_wins else team2} seems stronger recently."
    return response
