from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from backend.rag_system.utils import check_embeddings_status
from backend.rag_system.ingestion import prepare_ipl_documents, process_team_statistics
from backend.rag_system.rag_chain import run_rag
# from backend.rag_system.rag_chain import ask_gemini
from backend.models.predict_winner_score import predict_match_score
from backend.models.predict_winner import predict_winner
from backend.models.predict_player_stats import ask_gemini

app = FastAPI()


app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Serve static image files (analytics plots)
app.mount("/analytics", StaticFiles(directory="analytics/graphs"), name="analytics")

# Request body model
class MatchTeams(BaseModel):
    team1: str
    team2: str

# Root page (Main page)
@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

# Analytics Dashboard page
@app.get("/analytics")
def analytics_dashboard():
    return FileResponse("frontend/analytics.html")

# Predict match winner and scores

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
    
    
# Gemini-based player prediction
@app.get("/predict/player-performance/")
def get_player_prediction(player_name: str = Query(...), opponent_team: str = Query(...)):
    try:
        result = ask_gemini(player_name, opponent_team)
        return {"player": player_name, "opponent": opponent_team, "prediction": result}
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8003, reload=True)