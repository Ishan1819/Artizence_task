# ingestion.py

import pandas as pd
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import DATASET_PATH, VECTOR_STORE_PATH
from utils import check_embeddings_status, save_embeddings_status
import os
def prepare_ipl_documents():
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    current_year = datetime.now().year
    cutoff_year = current_year - 1
    df = df[df['year'] >= cutoff_year]

    documents = []
    for idx, row in df.iterrows():
        content = (
            f"Match ID: {idx}, Year: {row['year']}, "
            f"{row['team1']} scored {row['team1_score']} against {row['team2']} with a score of {row['team2_score']}. "
        )
        metadata = {
            "match_id": str(idx),
            "year": str(row['year']),
            "team1": row['team1'],
            "team2": row['team2'],
            "winner": row['winner'],
            "team1_score": str(row['team1_score']),
            "team2_score": str(row['team2_score']),
            # "venue": str(row.get('venue', 'unknown'))
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def process_team_statistics(df):
    team_stats = {}
    all_teams = set(df['team1'].unique()).union(set(df['team2'].unique()))

    for team in all_teams:
        team_matches = df[(df['team1'] == team) | (df['team2'] == team)]
        wins = len(team_matches[team_matches['winner'] == team])
        scores = [match['team1_score'] if match['team1'] == team else match['team2_score'] for _, match in team_matches.iterrows()]
        avg_score = sum(scores) / len(scores) if scores else 0

        team_stats[team] = {
            "matches": len(team_matches),
            "wins": wins,
            "win_rate": wins / len(team_matches) if team_matches is not None else 0,
            "avg_score": avg_score
        }

    stats_text = "IPL Team Statistics Analysis:\n\n"
    for team, stats in team_stats.items():
        stats_text += f"Team: {team}\nMatches: {stats['matches']}\nWins: {stats['wins']}\n"
        stats_text += f"Win Rate: {stats['win_rate']*100:.1f}%\nAverage Score: {stats['avg_score']:.1f}\n\n"

    stats_text += "Head-to-Head Analysis:\n\n"
    teams_list = list(all_teams)
    for i in range(len(teams_list)):
        for j in range(i+1, len(teams_list)):
            team1, team2 = teams_list[i], teams_list[j]
            h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) |
                     ((df['team1'] == team2) & (df['team2'] == team1))]
            if len(h2h) > 0:
                stats_text += f"{team1} vs {team2}:\n"
                stats_text += f"{team1} wins: {len(h2h[h2h['winner'] == team1])}\n"
                stats_text += f"{team2} wins: {len(h2h[h2h['winner'] == team2])}\n\n"
    return stats_text

def ingest_ipl_data():
    if check_embeddings_status() and os.path.exists(VECTOR_STORE_PATH):
        print(f"✅ Embeddings already exist at {VECTOR_STORE_PATH}. Skipping ingestion.")
        return

    print("Creating new embeddings...")
    match_documents = prepare_ipl_documents()

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df = df[df['year'] >= datetime.now().year - 1]

    stats_text = process_team_statistics(df)
    stats_doc = Document(page_content=stats_text, metadata={"type": "statistics", "year": str(datetime.now().year)})

    all_docs = match_documents + [stats_doc]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=VECTOR_STORE_PATH)

    save_embeddings_status(True)
    print("✅ Embeddings created and saved.")
