# from fastapi import FastAPI
# from pydantic import BaseModel


# app = FastAPI()

# class MatchTeams(BaseModel):
#     team1: str
#     team2: str

# @app.post("/predict-winner")
# def get_winner(data: MatchTeams):
#     result = predict_winner(data.team1, data.team2)
#     return result


# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the IPL Prediction API"}


from fastapi import FastAPI
from pydantic import BaseModel
from backend.models.predict_winner_score import predict_match_score  # Adjust this path to the actual location
from backend.models.predict_winner import predict_winner
from backend.models.predict_player_stats import ask_gemini  # Adjust this path to the actual location
from fastapi import Query
app = FastAPI()

class MatchTeams(BaseModel):
    team1: str
    team2: str

@app.post("/predict-winner-score")
def get_winner_score(data: MatchTeams):
    result = predict_match_score(data.team1, data.team2)
    return result


@app.post("/predict-winner")
def get_winner(data: MatchTeams):
    result = predict_winner(data.team1, data.team2)
    return result
# @app.post("/predict-score")
# def get_loser_score(data: MatchTeams):
#     result = predict_match_score(data.team1, data.team2)
#     return {"predicted_loser_score": result["predicted_loser_score"]}
@app.get("/predict/player-performance/")
def get_player_prediction(player_name: str = Query(...), opponent_team: str = Query(...)):
    try:
        result = ask_gemini(player_name, opponent_team)
        return {
            "player": player_name,
            "opponent": opponent_team,
            "prediction": result
        }
    except Exception as e:
        return {
            "error": str(e)
        }


@app.get("/")
def read_root():
    return {"message": "Welcome to the IPL Prediction API"}
