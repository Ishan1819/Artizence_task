# # from fastapi import FastAPI
# # from pydantic import BaseModel


# # app = FastAPI()

# # class MatchTeams(BaseModel):
# #     team1: str
# #     team2: str

# # @app.post("/predict-winner")
# # def get_winner(data: MatchTeams):
# #     result = predict_winner(data.team1, data.team2)
# #     return result


# # @app.get("/")
# # def read_root():
# #     return {"message": "Welcome to the IPL Prediction API"}


# from fastapi import FastAPI
# from pydantic import BaseModel

# # from backend.models.predict_winner_score import predict_match_score
# from backend.models.predict_winner_score import (
#     predict_match_score,
# )  # Adjust this path to the actual location
# from backend.models.predict_winner import predict_winner

# # from backend.models.predict_winner import predict_winner
# from backend.models.predict_player_stats import (
#     ask_gemini,
# )  # Adjust this path to the actual location
# from fastapi import Query
# from fastapi.staticfiles import StaticFiles


# app = FastAPI()

# app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")
# class MatchTeams(BaseModel):
#     team1: str
#     team2: str


# @app.post("/predict-winner-score")
# def get_winner_score(data: MatchTeams):
#     result = predict_match_score(data.team1, data.team2)
#     return result


# @app.post("/predict-winner")
# def get_combined_prediction(data: MatchTeams):
#     winner_result = predict_winner(data.team1, data.team2)

#     # Ensure prediction is valid
#     if "error" in winner_result:
#         return winner_result

#     # Call score prediction function
#     score_result = predict_match_score(data.team1, data.team2)

#     # Ensure winner gets the higher score
#     if winner_result["predicted_winner"] == score_result["team1"]:
#         winner_score = score_result["estimated_winner_score"]
#         loser_score = score_result["predicted_loser_score"]
#     else:
#         winner_score = score_result["estimated_winner_score"]
#         loser_score = score_result["predicted_loser_score"]
#         # Swap team1/team2 if needed
#         score_result["team1"], score_result["team2"] = (
#             score_result["team2"],
#             score_result["team1"],
#         )

#     return {
#         "team1": data.team1,
#         "team2": data.team2,
#         "predicted_winner": winner_result["predicted_winner"],
#         "probability": winner_result["probability"],
#         "team1_strength": winner_result["team1_strength"],
#         "team2_strength": winner_result["team2_strength"],
#         "explanation": winner_result["explanation"],
#         "predicted_scores": {
#             winner_result["predicted_winner"]: int(round(winner_score)),
#             (
#                 data.team1
#                 if data.team1 != winner_result["predicted_winner"]
#                 else data.team2
#             ): int(round(loser_score)),
#         },
#         "estimated_margin": abs(round(winner_score - loser_score)),
#     }


# @app.get("/predict/player-performance/")
# def get_player_prediction(
#     player_name: str = Query(...), opponent_team: str = Query(...)
# ):
#     try:
#         result = ask_gemini(player_name, opponent_team)
#         return {"player": player_name, "opponent": opponent_team, "prediction": result}
#     except Exception as e:
#         return {"error": str(e)}


# from fastapi.responses import FileResponse

# @app.get("/")
# def read_root():
#     return FileResponse("frontend/index.html")



# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)


from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.models.predict_winner_score import predict_match_score
from backend.models.predict_winner import predict_winner
from backend.models.predict_player_stats import ask_gemini

from fastapi import Query

app = FastAPI()

# Mount static files
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

class MatchTeams(BaseModel):
    team1: str
    team2: str

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

@app.post("/predict-winner")
def get_combined_prediction(data: MatchTeams):
    try:
        # Call the predict_winner function
        winner_result = predict_winner(data.team1, data.team2)
        if "error" in winner_result:
            return {"error": "Prediction failed", "details": winner_result}

        # Call the predict_match_score function
        score_result = predict_match_score(data.team1, data.team2)

        # Ensure winner gets the higher score
        if winner_result["predicted_winner"] == score_result["team1"]:
            winner_score = score_result["estimated_winner_score"]
            loser_score = score_result["predicted_loser_score"]
        else:
            winner_score = score_result["estimated_winner_score"]
            loser_score = score_result["predicted_loser_score"]
            score_result["team1"], score_result["team2"] = score_result["team2"], score_result["team1"]

        return {
            "team1": data.team1,
            "team2": data.team2,
            "predicted_winner": winner_result["predicted_winner"],
            "probability": winner_result["probability"],
            "team1_strength": winner_result["team1_strength"],
            "team2_strength": winner_result["team2_strength"],
            "explanation": winner_result["explanation"],
            "predicted_scores": {
                winner_result["predicted_winner"]: int(round(winner_score)),
                (
                    data.team1
                    if data.team1 != winner_result["predicted_winner"]
                    else data.team2
                ): int(round(loser_score)),
            },
            "estimated_margin": abs(round(winner_score - loser_score)),
        }
    except Exception as e:
        return {"error": "An error occurred while processing the prediction", "details": str(e)}

@app.get("/predict/player-performance/")
def get_player_prediction(player_name: str = Query(...), opponent_team: str = Query(...)):
    try:
        result = ask_gemini(player_name, opponent_team)
        return {"player": player_name, "opponent": opponent_team, "prediction": result}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)