# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime

# # Load the data
# df = pd.read_csv("D:/Ishan_ip datasets/merged_with_year.csv")

# # Filter to only include data from the last 1 year dynamically
# current_year = datetime.now().year
# cutoff_year = current_year - 1
# df = df[df['year'] >= cutoff_year]

# # Keep only necessary columns for team winner prediction
# team_prediction_df = df[['team1', 'team2', 'winner']].copy()
# team_prediction_df = team_prediction_df.dropna(subset=['team1', 'team2', 'winner'])

# # Print some basic stats
# print(f"Dataset contains {len(team_prediction_df)} matches from last one year onwards")
# print(f"Unique teams: {team_prediction_df['team1'].nunique()}")

# # Encode team names
# team_encoder = LabelEncoder()
# all_teams = pd.concat([team_prediction_df['team1'], team_prediction_df['team2'], team_prediction_df['winner']]).unique()
# team_encoder.fit(all_teams)

# team_prediction_df['team1_encoded'] = team_encoder.transform(team_prediction_df['team1'])
# team_prediction_df['team2_encoded'] = team_encoder.transform(team_prediction_df['team2'])
# team_prediction_df['winner_encoded'] = team_encoder.transform(team_prediction_df['winner'])

# # Binary target: whether team1 won
# team_prediction_df['team1_won'] = (team_prediction_df['team1'] == team_prediction_df['winner']).astype(int)

# # Compute historical win percentage as team strength
# team_stats = {}
# for team in all_teams:
#     team1_matches = team_prediction_df[team_prediction_df['team1'] == team]
#     team2_matches = team_prediction_df[team_prediction_df['team2'] == team]
#     team1_wins = sum(team1_matches['team1'] == team1_matches['winner'])
#     team2_wins = sum(team2_matches['team2'] == team2_matches['winner'])
#     total_matches = len(team1_matches) + len(team2_matches)
#     win_pct = (team1_wins + team2_wins) / total_matches if total_matches > 0 else 0.5
#     team_stats[team] = win_pct

# team_prediction_df['team1_strength'] = team_prediction_df['team1'].map(team_stats)
# team_prediction_df['team2_strength'] = team_prediction_df['team2'].map(team_stats)
# team_prediction_df['strength_diff'] = team_prediction_df['team1_strength'] - team_prediction_df['team2_strength']

# # Define features and labels
# X = team_prediction_df[['team1_encoded', 'team2_encoded', 'strength_diff']]
# y = team_prediction_df['team1_won']

# # Split into train/test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# # Random Forest model instead of XGBoost
# model = RandomForestClassifier(
#     n_estimators=150,
#     max_depth=7,
#     random_state=42
# )
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f"Model Accuracy: {accuracy:.4f}")
# print("Classification Report:")
# print(report)

# # Feature importance
# importance_df = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': model.feature_importances_
# }).sort_values('Importance', ascending=False)

# print("\nFeature Importance:")
# print(importance_df)

# # Sample Prediction
# team1 = "Gujarat Titans"
# team2 = "Chennai Super Kings"

# try:
#     team1_encoded = team_encoder.transform([team1])[0]
#     team2_encoded = team_encoder.transform([team2])[0]
#     team1_strength = team_stats.get(team1, 0.5)
#     team2_strength = team_stats.get(team2, 0.5)
#     strength_diff = team1_strength - team2_strength

#     team1_win_prob = model.predict_proba([[team1_encoded, team2_encoded, strength_diff]])[0][1]

#     if team1_win_prob > 0.5:
#         predicted_winner = team1
#         win_probability = team1_win_prob
#     else:
#         predicted_winner = team2
#         win_probability = 1 - team1_win_prob

#     print(f"\nPrediction for {team1} vs {team2}:")
#     print(f"Team strengths: {team1}: {team1_strength:.4f}, {team2}: {team2_strength:.4f}")
#     print(f"Predicted winner: {predicted_winner} with {win_probability:.2%} probability")
# except ValueError:
#     print(f"Error: One or both teams ({team1}, {team2}) not found in training data")


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBClassifier, XGBRegressor
# from sklearn.model_selection import train_test_split
# from datetime import datetime

# # ----------------------------------------------------------------------------
# # 1. TRAIN / PREPARE
# # ----------------------------------------------------------------------------
# df = pd.read_csv('D:/Ishan_ip datasets/merged_with_year.csv')

# # --- Filter data for the last year (current year - 1) ---
# current_year = datetime.now().year
# cutoff_year = current_year - 1
# df = df[df['year'] >= cutoff_year]

# # --- a) Winner classifier prep ---
# team_df = df[['team1','team2','winner']].dropna()
# all_teams = pd.concat([team_df['team1'],team_df['team2'],team_df['winner']]).unique()
# team_enc = LabelEncoder().fit(all_teams)

# team_df['t1_enc'] = team_enc.transform(team_df['team1'])
# team_df['t2_enc'] = team_enc.transform(team_df['team2'])
# team_df['t1_won'] = (team_df['team1']==team_df['winner']).astype(int)

# # historical strength feature
# win_pct = {}
# for t in all_teams:
#     m1 = team_df[team_df['team1']==t]
#     m2 = team_df[team_df['team2']==t]
#     w1 = (m1['team1']==m1['winner']).sum()
#     w2 = (m2['team2']==m2['winner']).sum()
#     total = len(m1)+len(m2)
#     win_pct[t] = (w1+w2)/total if total>0 else 0.5

# team_df['s1'] = team_df['team1'].map(win_pct)
# team_df['s2'] = team_df['team2'].map(win_pct)
# team_df['sdiff'] = team_df['s1']-team_df['s2']

# Xw = team_df[['t1_enc','t2_enc','sdiff']]
# yw = team_df['t1_won']
# Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(Xw,yw,test_size=0.25, random_state=42)

# clf = XGBClassifier(
#     n_estimators=200, learning_rate=0.1, max_depth=5,
#     subsample=0.8, colsample_bytree=0.8,
#     objective='binary:logistic', random_state=42
# )
# clf.fit(Xw_tr, yw_tr)

# # --- b) Score regressor prep (predicting loser_score) ---
# score_df = df[['team1','team2','winner','winner_score','loser_score']].dropna()
# score_df['winner_score'] = pd.to_numeric(score_df['winner_score'])
# score_df['loser_score'] = pd.to_numeric(score_df['loser_score'])

# # encode teams for the regressor
# reg_enc = LabelEncoder().fit(pd.concat([score_df['team1'],score_df['team2']]).unique())
# score_df['t1_enc']   = reg_enc.transform(score_df['team1'])
# score_df['t2_enc']   = reg_enc.transform(score_df['team2'])
# score_df['win_enc']  = reg_enc.transform(score_df['winner'])
# score_df['t1_won']   = (score_df['team1']==score_df['winner']).astype(int)

# # compute each team’s historical average winning total
# avg_win_score = score_df.groupby('winner')['winner_score'].mean().to_dict()

# # features: team1, team2, winner, flag, winner_score
# Xr = score_df[['t1_enc','t2_enc','win_enc','t1_won','winner_score']]
# yr = score_df['loser_score']
# Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr,yr,test_size=0.25,random_state=42)

# reg = XGBRegressor(
#     n_estimators=200, learning_rate=0.01, max_depth=5,
#     subsample=0.8, colsample_bytree=0.8,
#     objective='reg:squarederror', random_state=42
# )
# reg.fit(Xr_tr, yr_tr)

# # ----------------------------------------------------------------------------
# # 2. INFERENCE FUNCTION
# # ----------------------------------------------------------------------------
# def predict_match_score(team1, team2):
#     # --- a) encode & strength ---
#     try:
#         t1c = team_enc.transform([team1])[0]
#         t2c = team_enc.transform([team2])[0]
#     except ValueError:
#         raise ValueError(f"One of the teams not in training data: {team1}, {team2}")

#     s1 = win_pct.get(team1, 0.5)
#     s2 = win_pct.get(team2, 0.5)
#     sdiff = s1 - s2

#     # --- b) predict winner ---
#     prob1 = clf.predict_proba([[t1c, t2c, sdiff]])[0,1]
#     if prob1 > 0.5:
#         winner, loser = team1, team2
#         win_prob = prob1
#         t1_won_flag = 1
#     else:
#         winner, loser = team2, team1
#         win_prob = 1-prob1
#         t1_won_flag = 0

#     # --- c) estimate winner’s score (use historical average) ---
#     est_win_score = avg_win_score.get(winner, score_df['winner_score'].mean())

#     # --- d) assemble regressor features ---
#     # note: reg_enc is for the regressor, not the classifier
#     t1c_r = reg_enc.transform([team1])[0]
#     t2c_r = reg_enc.transform([team2])[0]
#     w_enc = reg_enc.transform([winner])[0]

#     Xreg = np.array([[t1c_r, t2c_r, w_enc, t1_won_flag, est_win_score]])
#     est_loser_score = reg.predict(Xreg)[0]

#     return {
#         'team1': team1,
#         'team2': team2,
#         'predicted_winner': winner,
#         'win_probability': win_prob,
#         'estimated_winner_score': est_win_score,
#         'predicted_loser_score': round(est_loser_score),
#         'estimated_margin': est_win_score - round(est_loser_score)
#     }

# # ----------------------------------------------------------------------------
# # 3. TRY IT OUT
# # ----------------------------------------------------------------------------
# out = predict_match_score("Delhi Capitals",
#                           "Kolkata Knight Riders")
# print(out)

# # Sample output might look like:
# # {
# #   'team1': 'Royal Challengers Bangalore',
# #   'team2': 'Chennai Super Kings',
# #   'predicted_winner': 'Chennai Super Kings',
# #   'win_probability': 0.67,
# #   'estimated_winner_score': 175.3,
# #   'predicted_loser_score': 158,
# #   'estimated_margin': 17.3
# # }


# ----------------------------------------------------------------------------------
# import google.generativeai as genai

# # Configure Gemini
# genai.configure(api_key="AIzaSyCYYUDOTqdhMC_NDbrQS-htFND7vocAIes")  # Replace with your Gemini API key

# model = genai.GenerativeModel("models/gemini-1.5-flash")

# # System-style prompt template
# def get_player_stats(player_name, opponent_team_name):
#     prompt = f"""
# You are an expert cricket analyst. Provide the detailed stats for the IPL player '{player_name}'.
# Give me all the bellow statistics strictly as given in numbers only. Remember that if the prediction is about a batsman or batter then give Predicted Wickets to be taken as 0 as he is a bowler and can't bowl. No words or sentences should be written. Also if you don't have the data, then write "N/A" in the field. Also if you want some date range then take as you want and display the data.
# Strictly follow this structured format:

# Player Name: <player_name>
# Team: <team_name>
# Opponent Team: <opponent_team_name>
# Predicted Score: <number>
# Strike Rate: <number>
# Predicted Wickets to be taken: <number>
# Economy Rate: <number>
# Runs to be concede: <number>
# """

#     response = model.generate_content(prompt)
#     return response.text

# # Example usage
# player_name = input("Enter IPL player name: ")
# opponent_team_name = input("Enter opponent team name: ")
# stats = get_player_stats(player_name, opponent_team_name)
# print(stats)


# -----------------------------------------------------------------------------------
# import os
# import google.generativeai as genai

# # 🔑 Set your Gemini API Key
# genai.configure(api_key="AIzaSyCYYUDOTqdhMC_NDbrQS-htFND7vocAIes")

# # ✳️ Set your custom system prompt
# system_prompt = "You are a helpful chatbot."

# # 🌐 Initialize the Gemini model
# model = genai.GenerativeModel(
#     model_name="gemini-1.5-flash",
#     system_instruction=system_prompt
# )

# chat = model.start_chat(history=[])

# # 🚀 Start chatbot loop
# print("Gemini Chatbot is ready! Type 'exit' to quit.\n")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         print("Chatbot: Goodbye!")
#         break

#     try:
#         response = chat.send_message(user_input)
#         print("Chatbot:", response.text)
#     except Exception as e:
#         print("Error:", e)
#         print("Please try again.")


# -------------------------------------------------------------

# from river import compose
# from river import ensemble
# from river import metrics
# from river import preprocessing
# from river import stream
# from river import tree
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from datetime import datetime
# import time

# # Flag to track if new data has been added
# data_updated_flag = False

# # Function to update the CSV file with new data
# def add_new_data(new_data):
#     global data_updated_flag
#     # Append the new data to the CSV file
#     new_df = pd.DataFrame(new_data)
#     new_df.to_csv("D:/Ishan_ip datasets/merged_with_year.csv", mode="a", header=False, index=False)
#     data_updated_flag = True  # Set the flag to True to indicate new data is added

# # Function to train the model with the latest data
# def train_model():
#     global data_updated_flag

#     # Load the dataset
#     df = pd.read_csv("D:/Ishan_ip datasets/merged_with_year.csv")

#     # Filter last 1 year of data
#     current_year = datetime.now().year
#     df = df[df["year"] >= current_year - 1]

#     # Drop rows with nulls in important columns
#     df = df.dropna(subset=["team1", "team2", "winner"])

#     # Prepare target: 1 if team1 wins, else 0
#     df["team1_won"] = (df["team1"] == df["winner"]).astype(int)

#     # Convert DataFrame into stream of dictionaries
#     data = list(zip(
#         df[["team1", "team2"]].to_dict(orient="records"),
#         df["team1_won"]
#     ))

#     # Build pipeline using AdaBoostClassifier with decision tree as base model
#     model = compose.Pipeline(
#         preprocessing.OneHotEncoder(),
#         ensemble.AdaBoostClassifier(
#             model=tree.HoeffdingTreeClassifier(grace_period=50),
#             n_models=20,
#             seed=42
#         )
#     )

#     # Metric
#     metric = metrics.Accuracy()

#     # Lists to store true and predicted values for confusion matrix
#     y_true = []
#     y_pred = []

#     # Train incrementally
#     for x, y in data:
#         # Make a prediction
#         y_pred_val = model.predict_one(x)
#         # Learn from the instance
#         model.learn_one(x, y)
#         # Update metric
#         metric.update(y, y_pred_val)

#         # Store values for confusion matrix
#         y_true.append(int(y))
#         y_pred.append(int(y_pred_val) if y_pred_val is not None else 0)  # Default to 0 if None

#     # Print out the accuracy
#     print(f"Accuracy: {metric.get():.4f}")

#     # Create and display confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     class_names = ['Team 2 wins', 'Team 1 wins']
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

#     plt.figure(figsize=(8, 6))
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title('Confusion Matrix')
#     plt.savefig('confusion_matrix.png')
#     print("\nConfusion matrix saved as 'confusion_matrix.png'")

#     # Calculate and display additional metrics from confusion matrix
#     tn, fp, fn, tp = cm.ravel()
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#     print("\nAdditional metrics:")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")

#     # Reset the flag after retraining the model
#     data_updated_flag = False


# # Function to check the flag periodically and retrain the model if needed
# def check_for_new_data():
#     while True:
#         if data_updated_flag:
#             print("\nNew data detected, retraining model...")
#             train_model()
#         else:
#             print("\nNo new data. Waiting for new data...")

#         time.sleep(60)  # Check every minute (can be adjusted based on your needs)


# # Simulate adding new data
# new_data = [
#     {"team1": "Mumbai Indians", "team2": "Delhi Capitals", "winner": "Mumbai Indians", "year": 2025},
#     {"team1": "Rajasthan Royals", "team2": "Kolkata Knight Riders", "winner": "Rajasthan Royals", "year": 2025}
# ]

# # Add new data and trigger the retraining process
# add_new_data(new_data)

# # Start checking for new data and retrain if needed
# check_for_new_data()


# ______________________________________________________

# import pandas as pd
# import json

# # Load your CSV
# df = pd.read_csv("D:/Ishan_ip datasets/player_metrics.csv")

# # Create the JSONL file
# with open("fine_tune_data.jsonl", "w") as f:
#     for _, row in df.iterrows():
#         input_text = (
#             f"Player {row['player']} from {row['team']} scored {row['runs_scored']} runs in {row['balls_faced']} balls "
#             f"with run rate of {row['run_rate']} and took {row['wickets_taken']} wickets. "
#             f"His economy rate is {row['economy_rate']} and runs conceded are {row['runs_conceded']}, "
#             f"with number of 4s are {row['number_of_4s']} and number of sixes are {row['number_of_6s']}."
#         )

#         # You can design this based on what you want the model to predict. Here's a sample:
#         output_text = (
#             f"Given this performance, {row['player']} is showing strength in batting and bowling. "
#             f"Maintaining a run rate of {row['run_rate']} and an economy of {row['economy_rate']} "
#             f"makes him a balanced player. Keep an eye on his next match."
#         )

#         example = {
#             "input_text": input_text,
#             "output_text": output_text
#         }

#         f.write(json.dumps(example) + "\n")


# ----------------------------------------------------

import google.generativeai as genai
from typing import Dict, Any


def configure_gemini(api_key: str) -> None:
    """Configure the Gemini API with the provided key."""
    genai.configure(api_key="AIzaSyCYYUDOTqdhMC_NDbrQS-htFND7vocAIes")


def get_player_stats(player_name: str, opponent_team_name: str) -> Dict[str, Any]:
    """
    Query Gemini API for cricket player statistics and predictions.

    Args:
        player_name: Name of the cricket player
        opponent_team_name: Name of the opponent team

    Returns:
        Dictionary containing structured player statistics
    """
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")

        # Create a structured prompt for consistent formatting
        prompt = f"""
        As an expert cricket analyst, provide detailed IPL statistics for '{player_name}' against '{opponent_team_name}'.
        
        Return ONLY the structured data in this exact format with numerical values:
        
        Player Name: {player_name}
        Team: [current team]
        Opponent Team: {opponent_team_name}
        Predicted Score: [number]
        Strike Rate: [number]
        Predicted Wickets to be taken: [number]
        Economy Rate: [number]
        Runs to be conceded: [number]
        
        Important guidelines:
        - If the player is a batsman, set "Predicted Wickets to be taken" to 0
        - If data is unavailable for any field, use "N/A"
        - Provide only numbers, no explanatory text
        - Use your knowledge to determine the player's current team
        """

        # Generate response
        response = model.generate_content(prompt)

        if not response.text:
            return {"error": "No response received from API"}

        # Parse the response into a structured dictionary
        result = {}
        for line in response.text.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()

        return result

    except Exception as e:
        return {"error": f"Error getting player stats: {str(e)}"}


def main():
    """Main function to run the program."""
    print("IPL Player Statistics Predictor\n")

    # Use hardcoded API key
    api_key = "AIzaSyCYYUDOTqdhMC_NDbrQS-htFND7vocAIes"

    try:
        # Configure API
        configure_gemini(api_key)

        # Get player and opponent details
        player_name = input("\nEnter IPL player name: ").strip()
        opponent_team_name = input("Enter opponent team name: ").strip()

        if not player_name or not opponent_team_name:
            print("Error: Player name and opponent team name cannot be empty.")
            return

        print("\nFetching stats, please wait...\n")

        # Get stats
        stats = get_player_stats(player_name, opponent_team_name)

        # Check for errors
        if "error" in stats:
            print(f"Error: {stats['error']}")
            return

        # Print results in a formatted way
        print("=" * 40)
        print(f"Statistics for {player_name}")
        print("=" * 40)

        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
