# backend/models/predict_winner.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from backend.rag_system.rag_chain import run_rag  

model = None
team_encoder = None
team_stats = {}


def train_model():
    global model, team_encoder, team_stats

    df = pd.read_csv("D:/Ishan_ip datasets/merged_with_year.csv")
    current_year = datetime.now().year
    cutoff_year = current_year - 1
    df = df[df["year"] >= cutoff_year]

    team_prediction_df = df[["team1", "team2", "winner"]].dropna()

    team_encoder = LabelEncoder()
    all_teams = pd.concat(
        [
            team_prediction_df["team1"],
            team_prediction_df["team2"],
            team_prediction_df["winner"],
        ]
    ).unique()
    team_encoder.fit(all_teams)

    team_prediction_df["team1_encoded"] = team_encoder.transform(
        team_prediction_df["team1"]
    )
    team_prediction_df["team2_encoded"] = team_encoder.transform(
        team_prediction_df["team2"]
    )
    team_prediction_df["winner_encoded"] = team_encoder.transform(
        team_prediction_df["winner"]
    )

    team_prediction_df["team1_won"] = (
        team_prediction_df["team1"] == team_prediction_df["winner"]
    ).astype(int)

    for team in all_teams:
        team1_matches = team_prediction_df[team_prediction_df["team1"] == team]
        team2_matches = team_prediction_df[team_prediction_df["team2"] == team]
        team1_wins = sum(team1_matches["team1"] == team1_matches["winner"])
        team2_wins = sum(team2_matches["team2"] == team2_matches["winner"])
        total_matches = len(team1_matches) + len(team2_matches)
        win_pct = (
            (team1_wins + team2_wins) / total_matches if total_matches > 0 else 0.5
        )
        team_stats[team] = win_pct

    team_prediction_df["team1_strength"] = team_prediction_df["team1"].map(team_stats)
    team_prediction_df["team2_strength"] = team_prediction_df["team2"].map(team_stats)
    team_prediction_df["strength_diff"] = (
        team_prediction_df["team1_strength"] - team_prediction_df["team2_strength"]
    )

    X = team_prediction_df[["team1_encoded", "team2_encoded", "strength_diff"]]
    y = team_prediction_df["team1_won"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
    model.fit(X_train, y_train)


train_model() 


def predict_winner(team1: str, team2: str):
    try:
        team1_encoded = team_encoder.transform([team1])[0]
        team2_encoded = team_encoder.transform([team2])[0]
        team1_strength = team_stats.get(team1, 0.5)
        team2_strength = team_stats.get(team2, 0.5)
        strength_diff = team1_strength - team2_strength

        team1_win_prob = model.predict_proba(
            [[team1_encoded, team2_encoded, strength_diff]]
        )[0][1]

        if team1_win_prob > 0.5:
            predicted_winner = team1
            win_probability = team1_win_prob
        else:
            predicted_winner = team2
            win_probability = 1 - team1_win_prob

        # Call RAG
        prompt = (
            f"Why do you predict {predicted_winner} will win between {team1} and {team2}? "
            f"Provide a detailed explanation based on historical match data, team strengths, and any other factors."
        )
        explanation = run_rag(prompt)

        return {
            "team1": team1,
            "team2": team2,
            "predicted_winner": predicted_winner,
            "probability": round(win_probability * 100, 2),
            "team1_strength": round(team1_strength, 3),
            "team2_strength": round(team2_strength, 3),
            "explanation": explanation
        }

    except ValueError:
        return {
            "error": f"One or both teams ({team1}, {team2}) not found in training data"
        }
