import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the data
df = pd.read_csv("D:/Ishan_ip datasets/merged_with_year.csv")

# Filter to only include data from the last 1 year dynamically
current_year = datetime.now().year
cutoff_year = current_year - 1
df = df[df["year"] >= cutoff_year]

# Keep only necessary columns for team winner prediction
team_prediction_df = df[["team1", "team2", "winner"]].copy()
team_prediction_df = team_prediction_df.dropna(subset=["team1", "team2", "winner"])

# Print some basic stats
print(f"Dataset contains {len(team_prediction_df)} matches from last one year onwards")
print(f"Unique teams: {team_prediction_df['team1'].nunique()}")

# Encode team names
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

# Binary target: whether team1 won
team_prediction_df["team1_won"] = (
    team_prediction_df["team1"] == team_prediction_df["winner"]
).astype(int)

# Compute historical win percentage as team strength
team_stats = {}
for team in all_teams:
    team1_matches = team_prediction_df[team_prediction_df["team1"] == team]
    team2_matches = team_prediction_df[team_prediction_df["team2"] == team]
    team1_wins = sum(team1_matches["team1"] == team1_matches["winner"])
    team2_wins = sum(team2_matches["team2"] == team2_matches["winner"])
    total_matches = len(team1_matches) + len(team2_matches)
    win_pct = (team1_wins + team2_wins) / total_matches if total_matches > 0 else 0.5
    team_stats[team] = win_pct

team_prediction_df["team1_strength"] = team_prediction_df["team1"].map(team_stats)
team_prediction_df["team2_strength"] = team_prediction_df["team2"].map(team_stats)
team_prediction_df["strength_diff"] = (
    team_prediction_df["team1_strength"] - team_prediction_df["team2_strength"]
)

# Define features and labels
X = team_prediction_df[["team1_encoded", "team2_encoded", "strength_diff"]]
y = team_prediction_df["team1_won"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Random Forest model instead of XGBoost
model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Feature importance
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Sample Prediction
team1 = "Gujarat Titans"
team2 = "Chennai Super Kings"

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

    print(f"\nPrediction for {team1} vs {team2}:")
    print(
        f"Team strengths: {team1}: {team1_strength:.4f}, {team2}: {team2_strength:.4f}"
    )
    print(
        f"Predicted winner: {predicted_winner} with {win_probability:.2%} probability"
    )
except ValueError:
    print(f"Error: One or both teams ({team1}, {team2}) not found in training data")
