import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from imblearn.over_sampling import SMOTE

# from fastapi import FastAPI
# from pydantic import BaseModel

# ----------------------------------------------------------------------------
# 1. TRAIN / PREPARE
# ----------------------------------------------------------------------------
df = pd.read_csv("D:/Ishan_ip datasets/merged_with_year.csv")

# --- Filter data for the last year (current year - 1) ---
current_year = datetime.now().year
cutoff_year = current_year - 1
df = df[df["year"] >= cutoff_year]

team_df = df[["team1", "team2", "winner"]].dropna()
all_teams = pd.concat([team_df["team1"], team_df["team2"], team_df["winner"]]).unique()
team_enc = LabelEncoder().fit(all_teams)

team_df["t1_enc"] = team_enc.transform(team_df["team1"])
team_df["t2_enc"] = team_enc.transform(team_df["team2"])
team_df["t1_won"] = (team_df["team1"] == team_df["winner"]).astype(int)

# Historical strength feature
win_pct = {}
for t in all_teams:
    m1 = team_df[team_df["team1"] == t]
    m2 = team_df[team_df["team2"] == t]
    w1 = (m1["team1"] == m1["winner"]).sum()
    w2 = (m2["team2"] == m2["winner"]).sum()
    total = len(m1) + len(m2)
    win_pct[t] = (w1 + w2) / total if total > 0 else 0.5

team_df["s1"] = team_df["team1"].map(win_pct)
team_df["s2"] = team_df["team2"].map(win_pct)
team_df["sdiff"] = team_df["s1"] - team_df["s2"]

Xw = team_df[["t1_enc", "t2_enc", "sdiff"]]
yw = team_df["t1_won"]

# Check imbalance
minority_ratio = min(yw.mean(), 1 - yw.mean())
print(f"Minority class ratio: {minority_ratio:.2f}")

if minority_ratio < 0.4:
    print("Applying SMOTE due to class imbalance.")
    smote = SMOTE(random_state=42)
    Xw, yw = smote.fit_resample(Xw, yw)
else:
    print("Data is balanced. No need for SMOTE.")

Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(Xw, yw, test_size=0.25, random_state=42)

clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    random_state=42,
)
clf.fit(Xw_tr, yw_tr)

# --- b) Score regressor prep (predicting loser_score) ---
score_df = df[["team1", "team2", "winner", "winner_score", "loser_score"]].dropna()
score_df["winner_score"] = pd.to_numeric(score_df["winner_score"])
score_df["loser_score"] = pd.to_numeric(score_df["loser_score"])

# encode teams for the regressor
reg_enc = LabelEncoder().fit(pd.concat([score_df["team1"], score_df["team2"]]).unique())
score_df["t1_enc"] = reg_enc.transform(score_df["team1"])
score_df["t2_enc"] = reg_enc.transform(score_df["team2"])
score_df["win_enc"] = reg_enc.transform(score_df["winner"])
score_df["t1_won"] = (score_df["team1"] == score_df["winner"]).astype(int)

# compute each team’s historical average winning total
avg_win_score = score_df.groupby("winner")["winner_score"].mean().to_dict()

# features: team1, team2, winner, flag, winner_score
Xr = score_df[["t1_enc", "t2_enc", "win_enc", "t1_won", "winner_score"]]
yr = score_df["loser_score"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, random_state=42)

reg = XGBRegressor(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)
reg.fit(Xr_tr, yr_tr)


# ----------------------------------------------------------------------------
# 2. INFERENCE FUNCTION
# ----------------------------------------------------------------------------
def predict_match_score(team1, team2):
    # --- a) encode & strength ---
    try:
        t1c = team_enc.transform([team1])[0]
        t2c = team_enc.transform([team2])[0]
    except ValueError:
        raise ValueError(f"One of the teams not in training data: {team1}, {team2}")

    s1 = win_pct.get(team1, 0.5)
    s2 = win_pct.get(team2, 0.5)
    sdiff = s1 - s2

    # --- b) predict winner ---
    prob1 = clf.predict_proba([[t1c, t2c, sdiff]])[0, 1]
    if prob1 > 0.5:
        winner, loser = team1, team2
        win_prob = prob1
        t1_won_flag = 1
    else:
        winner, loser = team2, team1
        win_prob = 1 - prob1
        t1_won_flag = 0

    # --- c) estimate winner’s score (use historical average) ---
    est_win_score = avg_win_score.get(winner, score_df["winner_score"].mean())

    # --- d) assemble regressor features ---
    t1c_r = reg_enc.transform([team1])[0]
    t2c_r = reg_enc.transform([team2])[0]
    w_enc = reg_enc.transform([winner])[0]

    Xreg = np.array([[t1c_r, t2c_r, w_enc, t1_won_flag, est_win_score]])
    est_loser_score = reg.predict(Xreg)[0]

    return {
        "team1": str(team1),
        "team2": str(team2),
        "predicted_winner": str(winner),
        "win_probability": float(win_prob * 100),  # 👈 Convert np.float32 to float
        "estimated_winner_score": float(est_win_score),
        "predicted_loser_score": int(round(est_loser_score)),
        "estimated_margin": float(est_win_score - round(est_loser_score)),
    }
