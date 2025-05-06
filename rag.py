import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from datetime import datetime
import os
import json

# Define constants
VECTOR_STORE_PATH = "./ipl_chroma_db"
DATASET_PATH = "D:/Ishan_ip datasets/cleaned_unique_matches.csv"
CONFIG_FILE = "./ipl_embeddings_status.json"


def check_embeddings_status():
    """
    Check if embeddings have already been created
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config.get("embeddings_created", False)
        except:
            return False
    return False


def save_embeddings_status(status=True):
    """
    Save the status of embeddings creation
    """
    with open(CONFIG_FILE, "w") as f:
        json.dump({"embeddings_created": status}, f)


def prepare_ipl_documents():
    """
    Load IPL data from CSV and convert to documents for vectorization
    """
    # Load your IPL data
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    # Filter for recent data
    current_year = datetime.now().year
    cutoff_year = current_year - 1
    df = df[df["year"] >= cutoff_year]

    # Convert to text chunks, creating richer context
    documents = []

    # Process each match into a document
    for idx, row in df.iterrows():
        # Create match content
        content = (
            f"Match ID: {idx}, Year: {row['year']}, "
            f"{row['team1']} scored {row['team1_score']} against {row['team2']} with a score of {row['team2_score']}. "
            f"Winner: {row['winner']}. "
            f"The match was played at {row.get('venue', 'unknown venue')}."
        )

        # Create enhanced metadata for better retrieval
        metadata = {
            "match_id": str(idx),
            "year": str(row["year"]),
            "team1": row["team1"],
            "team2": row["team2"],
            "winner": row["winner"],
            "team1_score": str(row["team1_score"]),
            "team2_score": str(row["team2_score"]),
            "venue": str(row.get("venue", "unknown")),
        }

        # Create LangChain document
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"Created {len(documents)} match documents")
    return documents


def process_team_statistics(df):
    """
    Process team statistics from dataframe, returning analysis text
    """
    # Create a dictionary of team stats
    team_stats = {}

    # Get all unique teams
    all_teams = set(df["team1"].unique()).union(set(df["team2"].unique()))

    for team in all_teams:
        # Get matches where this team played
        team_matches = df[(df["team1"] == team) | (df["team2"] == team)]

        # Calculate wins
        wins = len(team_matches[team_matches["winner"] == team])

        # Calculate average scores
        team_scores = []
        for _, match in team_matches.iterrows():
            if match["team1"] == team:
                team_scores.append(match["team1_score"])
            else:
                team_scores.append(match["team2_score"])

        avg_score = sum(team_scores) / len(team_scores) if team_scores else 0

        # Store stats
        team_stats[team] = {
            "matches": len(team_matches),
            "wins": wins,
            "win_rate": wins / len(team_matches) if len(team_matches) > 0 else 0,
            "avg_score": avg_score,
        }

    # Create a text document with all team stats
    stats_text = "IPL Team Statistics Analysis:\n\n"

    for team, stats in team_stats.items():
        stats_text += f"Team: {team}\n"
        stats_text += f"Matches: {stats['matches']}\n"
        stats_text += f"Wins: {stats['wins']}\n"
        stats_text += f"Win Rate: {stats['win_rate']*100:.1f}%\n"
        stats_text += f"Average Score: {stats['avg_score']:.1f}\n\n"

    # Add head-to-head analysis
    stats_text += "Head-to-Head Analysis:\n\n"

    teams_list = list(all_teams)
    for i in range(len(teams_list)):
        for j in range(i + 1, len(teams_list)):
            team1 = teams_list[i]
            team2 = teams_list[j]

            # Get matches between these teams
            h2h_matches = df[
                ((df["team1"] == team1) & (df["team2"] == team2))
                | ((df["team1"] == team2) & (df["team2"] == team1))
            ]

            if len(h2h_matches) > 0:
                team1_wins = len(h2h_matches[h2h_matches["winner"] == team1])
                team2_wins = len(h2h_matches[h2h_matches["winner"] == team2])

                stats_text += f"{team1} vs {team2}:\n"
                stats_text += f"{team1} wins: {team1_wins}\n"
                stats_text += f"{team2} wins: {team2_wins}\n\n"

    return stats_text


def ingest_ipl_data():
    """
    Ingest IPL data into Chroma vector store
    """
    # First check if embeddings already exist
    if check_embeddings_status() and os.path.exists(VECTOR_STORE_PATH):
        print(
            f"✅ Embeddings already exist at {VECTOR_STORE_PATH}. Skipping ingestion."
        )
        return

    print("Creating new embeddings. This may take a few minutes...")

    # Load and prepare documents
    match_documents = prepare_ipl_documents()

    # Load data for statistics
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    current_year = datetime.now().year
    cutoff_year = current_year - 1
    df = df[df["year"] >= cutoff_year]

    # Process team statistics into a document
    stats_text = process_team_statistics(df)
    stats_doc = Document(
        page_content=stats_text,
        metadata={"type": "statistics", "year": str(current_year)},
    )

    # Add stats document to our collection
    all_documents = match_documents + [stats_doc]

    # Split documents for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Split {len(all_documents)} documents into {len(chunks)} chunks.")

    # Initialize embedding model (using HuggingFace embeddings)
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Create Chroma vector store
    Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=VECTOR_STORE_PATH
    )

    # Save the status that embeddings have been created
    save_embeddings_status(True)

    print(f"✅ IPL data successfully indexed in {VECTOR_STORE_PATH}")


def create_ipl_rag_chain():
    """
    Create a RAG chain for IPL analysis
    """
    # Initialize the LLM
    model = ChatOllama(model="iplguru")

    # Create prompt template for IPL analysis
    prompt = PromptTemplate.from_template(
        """
        <s> [INST] <<SYS>>
        You are an expert IPL cricket analyst chatbot made by Ishan Patil. 
        Always give short and crisp answers about cricket. Prefer 2-3 bullet points or under 60 words. 
        Avoid long paragraphs or unnecessary detail. Your task is to reason through the question given by analyzing recent matches of the teams mentioned.
        
        Keep these guidelines in mind:
        - Start with a direct, confident answer
        - Include key statistics from the match data
        - Use cricket terminology appropriately
        - Be enthusiastic but concise
        
        Example responses:
        
        Q: How did you predict that Mumbai Indians will win today against Chennai Super Kings?
        A:
        • Mumbai Indians have won 4 of their last 5 matches, while CSK only won 2.
        • MI's batting lineup has been more consistent, averaging 175 runs compared to CSK's 160.
        • Head-to-head, MI has dominated CSK in recent encounters, winning 3 of their last 4 meetings.
        
        Q: How did you know that today Hyderabad will lose against GT?
        A:
        • GT have won 3 of their last 5 matches compared to SRH's 1 win.
        • GT's batting has been exceptional, averaging 170+ runs per match.
        • SRH's bowling attack has struggled against powerful batting lineups like GT's.
        <</SYS>>
        
        Use the following IPL match data and analysis to justify your answer:
        
        {context}
        
        Question: {input} [/INST]
        """
    )

    # Load vector store
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH, embedding_function=embedding
    )

    # Create retriever with specific search parameters
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,  # Retrieve 5 most relevant chunks
            "score_threshold": 0.5,  # Only include if relevance is > 0.5
        },
    )

    # Create document chain and retrieval chain
    document_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain


def fallback_response(question):
    """
    Generate a fallback response when the LLM fails
    """
    # Check for teams in the question
    ipl_teams = {
        "MI": "Mumbai Indians",
        "CSK": "Chennai Super Kings",
        "RCB": "Royal Challengers Bangalore",
        "KKR": "Kolkata Knight Riders",
        "DC": "Delhi Capitals",
        "SRH": "Sunrisers Hyderabad",
        "PBKS": "Punjab Kings",
        "RR": "Rajasthan Royals",
        "GT": "Gujarat Titans",
        "LSG": "Lucknow Super Giants",
    }

    # Add full names as values with themselves as keys
    for team_name in list(ipl_teams.values()):
        ipl_teams[team_name] = team_name

    mentioned_teams = []
    for team_abbr, team_name in ipl_teams.items():
        if team_abbr in question:
            if team_name not in mentioned_teams:
                mentioned_teams.append(team_name)

    if len(mentioned_teams) >= 2:
        team1, team2 = mentioned_teams[0], mentioned_teams[1]

        # Load data for statistics
        df = pd.read_csv(DATASET_PATH, low_memory=False)

        # Filter for recent matches
        current_year = datetime.now().year
        cutoff_year = current_year - 1
        df = df[df["year"] >= cutoff_year]

        # Get team1 stats
        team1_matches = df[(df["team1"] == team1) | (df["team2"] == team1)].tail(5)
        team1_wins = len(team1_matches[team1_matches["winner"] == team1])

        # Get team2 stats
        team2_matches = df[(df["team1"] == team2) | (df["team2"] == team2)].tail(5)
        team2_wins = len(team2_matches[team2_matches["winner"] == team2])

        # Get head-to-head
        h2h_matches = df[
            ((df["team1"] == team1) & (df["team2"] == team2))
            | ((df["team1"] == team2) & (df["team2"] == team1))
        ].tail(3)
        team1_h2h_wins = len(h2h_matches[h2h_matches["winner"] == team1])
        team2_h2h_wins = len(h2h_matches[h2h_matches["winner"] == team2])

        # Generate fallback response
        response = f"• In the past 5 matches, {team1} has won {team1_wins}, while {team2} has won {team2_wins} games.\n"

        # Determine stronger team for second bullet
        stronger_team = team1 if team1_wins > team2_wins else team2
        weaker_team = team2 if stronger_team == team1 else team1

        response += f"• {stronger_team}'s batting lineup has been in good form, showing consistency and strong performance.\n"
        response += f"• {weaker_team}'s bowling unit has struggled to contain powerful opposition batsmen, which could be exploited by {stronger_team}."

        return response
    else:
        return "• I need specific team names to provide accurate analysis.\n• Please mention the teams you're interested in for a detailed prediction.\n• Try asking about popular teams like MI, CSK, RCB, or GT."


def update_embeddings():
    """
    Function to manually update embeddings
    """
    # Delete existing embeddings
    if os.path.exists(VECTOR_STORE_PATH):
        import shutil

        shutil.rmtree(VECTOR_STORE_PATH)

    # Delete status file
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)

    # Create new embeddings
    ingest_ipl_data()
    print("✅ Embeddings updated successfully")


def main():
    """
    Main function to run the IPL RAG system
    """
    # Check if embeddings need to be created
    ingest_ipl_data()

    # Create RAG chain
    try:
        rag_chain = create_ipl_rag_chain()
        print("✅ RAG chain initialized successfully")
    except Exception as e:
        print(f"⚠️ Error initializing RAG chain: {e}")
        print("Proceeding with fallback mechanism only")
        rag_chain = None

    # Interactive loop
    while True:
        # Get user question
        question = input(
            "\nAsk your IPL question (or type 'exit' to quit, 'update' to refresh embeddings): "
        )

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        elif question.lower() == "update":
            update_embeddings()
            # Reinitialize RAG chain with new embeddings
            try:
                rag_chain = create_ipl_rag_chain()
                print("✅ RAG chain reinitialized successfully")
            except Exception as e:
                print(f"⚠️ Error reinitializing RAG chain: {e}")
                print("Proceeding with fallback mechanism only")
                rag_chain = None
            continue

        print("\n🏏 IPL Analysis:\n")

        try:
            # Try using the RAG chain if available
            if rag_chain:
                response = rag_chain.invoke({"input": question})
                answer = response.get("answer", "")

                # Check if we got a reasonable response
                if answer and len(answer) > 20:
                    print(answer)
                else:
                    # If response is too short or empty, use fallback
                    print(fallback_response(question))
            else:
                # Use fallback if RAG chain not available
                print(fallback_response(question))

        except Exception as e:
            print(f"Error generating response: {e}")
            print("\nFallback Analysis:\n")
            print(fallback_response(question))


if __name__ == "__main__":
    main()


# import pandas as pd
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.chat_models import ChatOllama
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.prompts import PromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain.schema import Document
# from datetime import datetime
# import os
# import json

# # Define constants
# VECTOR_STORE_PATH = "./ipl_chroma_db"
# DATASET_PATH = "./cleaned_unique_matches.csv"  # Adjusted path to be relative
# CONFIG_FILE = "./ipl_embeddings_status.json"


# def check_embeddings_status():
#     """
#     Check if embeddings have already been created
#     """
#     if os.path.exists(CONFIG_FILE):
#         try:
#             with open(CONFIG_FILE, 'r') as f:
#                 config = json.load(f)
#                 return config.get('embeddings_created', False)
#         except Exception as e:
#             print(f"Error reading config file: {e}")
#             return False
#     return False

# def save_embeddings_status(status=True):
#     """
#     Save the status of embeddings creation
#     """
#     try:
#         with open(CONFIG_FILE, 'w') as f:
#             json.dump({'embeddings_created': status}, f)
#     except Exception as e:
#         print(f"Error saving embeddings status: {e}")

# def prepare_ipl_documents():
#     """
#     Load IPL data from CSV and convert to documents for vectorization
#     """
#     # Load your IPL data
#     try:
#         df = pd.read_csv(DATASET_PATH, low_memory=False)
#     except FileNotFoundError:
#         print(f"Error: Dataset file not found at {DATASET_PATH}")
#         return []
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return []

#     # Filter for recent data
#     current_year = datetime.now().year
#     cutoff_year = current_year - 1
#     df = df[df['year'] >= cutoff_year]

#     # Convert to text chunks, creating richer context
#     documents = []

#     # Process each match into a document
#     for idx, row in df.iterrows():
#         try:
#             # Create match content with error handling for missing values
#             team1 = row.get('team1', 'unknown team')
#             team2 = row.get('team2', 'unknown team')
#             team1_score = row.get('team1_score', 0)
#             team2_score = row.get('team2_score', 0)
#             winner = row.get('winner', 'unknown')
#             venue = row.get('venue', 'unknown venue')
#             match_year = row.get('year', 'unknown year')

#             content = (
#                 f"Match ID: {idx}, Year: {match_year}, "
#                 f"{team1} scored {team1_score} against {team2} with a score of {team2_score}. "
#                 f"Winner: {winner}. "
#                 f"The match was played at {venue}."
#             )

#             # Create enhanced metadata for better retrieval
#             metadata = {
#                 "match_id": str(idx),
#                 "year": str(match_year),
#                 "team1": team1,
#                 "team2": team2,
#                 "winner": winner,
#                 "team1_score": str(team1_score),
#                 "team2_score": str(team2_score),
#                 "venue": venue
#             }

#             # Create LangChain document
#             doc = Document(page_content=content, metadata=metadata)
#             documents.append(doc)
#         except Exception as e:
#             print(f"Error processing row {idx}: {e}")
#             continue

#     print(f"Created {len(documents)} match documents")
#     return documents

# def process_team_statistics(df):
#     """
#     Process team statistics from dataframe, returning analysis text
#     """
#     # Create a dictionary of team stats
#     team_stats = {}

#     # Get all unique teams
#     all_teams = set()
#     if 'team1' in df.columns and 'team2' in df.columns:
#         all_teams = set(df['team1'].dropna().unique()).union(set(df['team2'].dropna().unique()))
#     else:
#         print("Warning: team1 or team2 columns not found in dataframe")
#         return "No team statistics available due to missing data."

#     for team in all_teams:
#         try:
#             # Get matches where this team played
#             team_matches = df[(df['team1'] == team) | (df['team2'] == team)]

#             # Calculate wins
#             if 'winner' in df.columns:
#                 wins = len(team_matches[team_matches['winner'] == team])
#             else:
#                 wins = 0
#                 print(f"Warning: winner column not found for team {team}")

#             # Calculate average scores
#             team_scores = []
#             for _, match in team_matches.iterrows():
#                 if match.get('team1') == team and 'team1_score' in match:
#                     team_scores.append(match['team1_score'])
#                 elif match.get('team2') == team and 'team2_score' in match:
#                     team_scores.append(match['team2_score'])

#             avg_score = sum(team_scores) / len(team_scores) if team_scores else 0

#             # Store stats
#             team_stats[team] = {
#                 "matches": len(team_matches),
#                 "wins": wins,
#                 "win_rate": wins / len(team_matches) if len(team_matches) > 0 else 0,
#                 "avg_score": avg_score
#             }
#         except Exception as e:
#             print(f"Error processing stats for team {team}: {e}")
#             continue

#     # Create a text document with all team stats
#     stats_text = "IPL Team Statistics Analysis:\n\n"

#     for team, stats in team_stats.items():
#         stats_text += f"Team: {team}\n"
#         stats_text += f"Matches: {stats['matches']}\n"
#         stats_text += f"Wins: {stats['wins']}\n"
#         stats_text += f"Win Rate: {stats['win_rate']*100:.1f}%\n"
#         stats_text += f"Average Score: {stats['avg_score']:.1f}\n\n"

#     # Add head-to-head analysis
#     stats_text += "Head-to-Head Analysis:\n\n"

#     teams_list = list(all_teams)
#     for i in range(len(teams_list)):
#         for j in range(i+1, len(teams_list)):
#             try:
#                 team1 = teams_list[i]
#                 team2 = teams_list[j]

#                 # Get matches between these teams
#                 h2h_matches = df[((df['team1'] == team1) & (df['team2'] == team2)) |
#                                 ((df['team1'] == team2) & (df['team2'] == team1))]

#                 if len(h2h_matches) > 0:
#                     team1_wins = len(h2h_matches[h2h_matches['winner'] == team1])
#                     team2_wins = len(h2h_matches[h2h_matches['winner'] == team2])

#                     stats_text += f"{team1} vs {team2}:\n"
#                     stats_text += f"{team1} wins: {team1_wins}\n"
#                     stats_text += f"{team2} wins: {team2_wins}\n\n"
#             except Exception as e:
#                 print(f"Error processing head-to-head for {teams_list[i]} vs {teams_list[j]}: {e}")
#                 continue

#     return stats_text

# def ingest_ipl_data():
#     """
#     Ingest IPL data into Chroma vector store
#     """
#     # First check if embeddings already exist
#     if check_embeddings_status() and os.path.exists(VECTOR_STORE_PATH):
#         print(f"✅ Embeddings already exist at {VECTOR_STORE_PATH}. Skipping ingestion.")
#         return

#     print("Creating new embeddings. This may take a few minutes...")

#     try:
#         # Load and prepare documents
#         match_documents = prepare_ipl_documents()

#         if not match_documents:
#             print("Error: No documents to embed. Check your dataset.")
#             return

#         # Load data for statistics
#         df = pd.read_csv(DATASET_PATH, low_memory=False)
#         current_year = datetime.now().year
#         cutoff_year = current_year - 1
#         df = df[df['year'] >= cutoff_year]

#         # Process team statistics into a document
#         stats_text = process_team_statistics(df)
#         stats_doc = Document(
#             page_content=stats_text,
#             metadata={"type": "statistics", "year": str(current_year)}
#         )

#         # Add stats document to our collection
#         all_documents = match_documents + [stats_doc]

#         # Split documents for better chunking
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1024,
#             chunk_overlap=100,
#             length_function=len,
#             add_start_index=True,
#         )
#         chunks = text_splitter.split_documents(all_documents)
#         print(f"Split {len(all_documents)} documents into {len(chunks)} chunks.")

#         # Check if chunks were created successfully
#         if not chunks:
#             print("Error: Document splitting resulted in no chunks.")
#             return

#         # Initialize embedding model (using HuggingFace embeddings)
#         try:
#             embedding = HuggingFaceEmbeddings(
#                 model_name="BAAI/bge-small-en-v1.5"
#             )
#         except Exception as e:
#             print(f"Error initializing embedding model: {e}")
#             print("Make sure you have the required dependencies installed:")
#             print("pip install sentence-transformers")
#             return

#         # Create Chroma vector store
#         try:
#             Chroma.from_documents(
#                 documents=chunks,
#                 embedding=embedding,
#                 persist_directory=VECTOR_STORE_PATH
#             )
#         except Exception as e:
#             print(f"Error creating Chroma vector store: {e}")
#             return

#         # Save the status that embeddings have been created
#         save_embeddings_status(True)

#         print(f"✅ IPL data successfully indexed in {VECTOR_STORE_PATH}")

#     except Exception as e:
#         print(f"Error during data ingestion: {e}")

# def create_ipl_rag_chain():
#     """
#     Create a RAG chain for IPL analysis
#     """
#     try:
#         # Initialize the LLM
#         try:
#             model = ChatOllama(model="iplguru")
#         except Exception as e:
#             print(f"Error initializing Ollama model: {e}")
#             print("Make sure Ollama is installed and running with the 'iplguru' model.")
#             print("Falling back to default model...")
#             # Fallback to default model
#             model = ChatOllama(model="llama2")

#         # Create prompt template for IPL analysis
#         prompt = PromptTemplate.from_template(
#             """
#             <s> [INST] <<SYS>>
#             You are an expert IPL cricket analyst chatbot made by Ishan Patil.
#             Always give short and crisp answers about cricket. Prefer 2-3 bullet points or under 60 words.
#             Avoid long paragraphs or unnecessary detail. Your task is to reason through the question given by analyzing recent matches of the teams mentioned.

#             Keep these guidelines in mind:
#             - Start with a direct, confident answer
#             - Include key statistics from the match data
#             - Use cricket terminology appropriately
#             - Be enthusiastic but concise

#             Example responses:

#             Q: How did you predict that Mumbai Indians will win today against Chennai Super Kings?
#             A:
#             • Mumbai Indians have won 4 of their last 5 matches, while CSK only won 2.
#             • MI's batting lineup has been more consistent, averaging 175 runs compared to CSK's 160.
#             • Head-to-head, MI has dominated CSK in recent encounters, winning 3 of their last 4 meetings.

#             Q: How did you know that today Hyderabad will lose against GT?
#             A:
#             • GT have won 3 of their last 5 matches compared to SRH's 1 win.
#             • GT's batting has been exceptional, averaging 170+ runs per match.
#             • SRH's bowling attack has struggled against powerful batting lineups like GT's.
#             <</SYS>>

#             Use the following IPL match data and analysis to justify your answer:

#             {context}

#             Question: {input} [/INST]
#             """
#         )

#         # Check if vector store exists
#         if not os.path.exists(VECTOR_STORE_PATH):
#             print(f"Error: Vector store not found at {VECTOR_STORE_PATH}")
#             print("Run ingest_ipl_data() first to create the vector store.")
#             return None

#         # Load vector store
#         try:
#             embedding = HuggingFaceEmbeddings(
#                 model_name="BAAI/bge-small-en-v1.5"
#             )

#             vector_store = Chroma(
#                 persist_directory=VECTOR_STORE_PATH,
#                 embedding_function=embedding
#             )
#         except Exception as e:
#             print(f"Error loading vector store: {e}")
#             return None

#         # Create retriever with specific search parameters
#         retriever = vector_store.as_retriever(
#             search_type="similarity_score_threshold",
#             search_kwargs={
#                 "k": 5,  # Retrieve 5 most relevant chunks
#                 "score_threshold": 0.5,  # Only include if relevance is > 0.5
#             },
#         )

#         # Create document chain and retrieval chain
#         document_chain = create_stuff_documents_chain(model, prompt)
#         rag_chain = create_retrieval_chain(retriever, document_chain)

#         return rag_chain

#     except Exception as e:
#         print(f"Error creating RAG chain: {e}")
#         return None

# def fallback_response(question):
#     """
#     Generate a fallback response when the LLM fails
#     """
#     try:
#         # Check for teams in the question
#         ipl_teams = {
#             "MI": "Mumbai Indians",
#             "CSK": "Chennai Super Kings",
#             "RCB": "Royal Challengers Bangalore",
#             "KKR": "Kolkata Knight Riders",
#             "DC": "Delhi Capitals",
#             "SRH": "Sunrisers Hyderabad",
#             "PBKS": "Punjab Kings",
#             "RR": "Rajasthan Royals",
#             "GT": "Gujarat Titans",
#             "LSG": "Lucknow Super Giants"
#         }

#         # Add full names as values with themselves as keys
#         for team_name in list(ipl_teams.values()):
#             ipl_teams[team_name] = team_name

#         mentioned_teams = []
#         for team_abbr, team_name in ipl_teams.items():
#             if team_abbr in question:
#                 if team_name not in mentioned_teams:
#                     mentioned_teams.append(team_name)

#         if len(mentioned_teams) >= 2:
#             try:
#                 # Try to load the dataset
#                 df = pd.read_csv(DATASET_PATH, low_memory=False)

#                 team1, team2 = mentioned_teams[0], mentioned_teams[1]

#                 # Filter for recent matches
#                 current_year = datetime.now().year
#                 cutoff_year = current_year - 1
#                 df = df[df['year'] >= cutoff_year]

#                 # Get team1 stats
#                 team1_matches = df[(df['team1'] == team1) | (df['team2'] == team1)].tail(5)
#                 team1_wins = len(team1_matches[team1_matches['winner'] == team1])

#                 # Get team2 stats
#                 team2_matches = df[(df['team1'] == team2) | (df['team2'] == team2)].tail(5)
#                 team2_wins = len(team2_matches[team2_matches['winner'] == team2])

#                 # Get head-to-head
#                 h2h_matches = df[((df['team1'] == team1) & (df['team2'] == team2)) |
#                                 ((df['team1'] == team2) & (df['team2'] == team1))].tail(3)
#                 team1_h2h_wins = len(h2h_matches[h2h_matches['winner'] == team1])
#                 team2_h2h_wins = len(h2h_matches[h2h_matches['winner'] == team2])

#                 # Generate fallback response
#                 response = f"• In the past 5 matches, {team1} has won {team1_wins}, while {team2} has won {team2_wins} games.\n"

#                 # Determine stronger team for second bullet
#                 stronger_team = team1 if team1_wins > team2_wins else team2
#                 weaker_team = team2 if stronger_team == team1 else team1

#                 response += f"• {stronger_team}'s batting lineup has been in good form, showing consistency and strong performance.\n"
#                 response += f"• {weaker_team}'s bowling unit has struggled to contain powerful opposition batsmen, which could be exploited by {stronger_team}."

#                 return response
#             except Exception as e:
#                 print(f"Error in fallback with data: {e}")
#                 # If data processing fails, provide a generic response
#                 return f"• Based on recent form, {team1} and {team2} have both shown strong performances.\n• Team chemistry and current momentum will be key factors in today's match.\n• Weather conditions and pitch factors could favor either team depending on their playing style."
#         else:
#             return "• I need specific team names to provide accurate analysis.\n• Please mention the teams you're interested in for a detailed prediction.\n• Try asking about popular teams like MI, CSK, RCB, or GT."
#     except Exception as e:
#         print(f"Error in fallback response: {e}")
#         return "• I'm unable to analyze the teams at the moment.\n• Please try again with a more specific question about IPL teams.\n• For best results, mention two specific teams by name."

# def update_embeddings():
#     """
#     Function to manually update embeddings
#     """
#     try:
#         # Delete existing embeddings
#         if os.path.exists(VECTOR_STORE_PATH):
#             import shutil
#             try:
#                 shutil.rmtree(VECTOR_STORE_PATH)
#                 print(f"Deleted existing vector store at {VECTOR_STORE_PATH}")
#             except Exception as e:
#                 print(f"Error deleting vector store: {e}")
#                 return

#         # Delete status file
#         if os.path.exists(CONFIG_FILE):
#             try:
#                 os.remove(CONFIG_FILE)
#                 print(f"Deleted configuration file at {CONFIG_FILE}")
#             except Exception as e:
#                 print(f"Error deleting config file: {e}")
#                 return

#         # Create new embeddings
#         ingest_ipl_data()
#         print("✅ Embeddings updated successfully")
#     except Exception as e:
#         print(f"Error updating embeddings: {e}")

# def check_dependencies():
#     """
#     Check if all required dependencies are installed
#     """
#     missing_dependencies = []

#     try:
#         import pandas
#     except ImportError:
#         missing_dependencies.append("pandas")

#     try:
#         from langchain_community.vectorstores import Chroma
#     except ImportError:
#         missing_dependencies.append("langchain-community")

#     try:
#         from langchain_community.embeddings import HuggingFaceEmbeddings
#     except ImportError:
#         missing_dependencies.append("sentence-transformers")

#     try:
#         from langchain_community.chat_models import ChatOllama
#     except ImportError:
#         missing_dependencies.append("langchain-community[ollama]")

#     if missing_dependencies:
#         print("⚠️ Missing dependencies detected. Please install the following packages:")
#         for dep in missing_dependencies:
#             print(f"  pip install {dep}")
#         return False

#     return True

# def check_dataset():
#     """
#     Check if the dataset file exists and is accessible
#     """
#     if not os.path.exists(DATASET_PATH):
#         print(f"⚠️ Dataset file not found at {DATASET_PATH}")
#         print("Please ensure the dataset file exists and is accessible.")
#         return False

#     try:
#         df = pd.read_csv(DATASET_PATH, nrows=5)  # Try reading just a few rows
#         required_columns = ['team1', 'team2', 'team1_score', 'team2_score', 'winner', 'year']
#         missing_columns = [col for col in required_columns if col not in df.columns]

#         if missing_columns:
#             print(f"⚠️ Dataset is missing required columns: {', '.join(missing_columns)}")
#             return False

#         return True
#     except Exception as e:
#         print(f"⚠️ Error reading dataset: {e}")
#         return False

# def main():
#     """
#     Main function to run the IPL RAG system
#     """
#     print("IPL Cricket Analysis RAG System")
#     print("===============================")

#     # Check dependencies
#     if not check_dependencies():
#         print("⚠️ Please install missing dependencies before continuing.")
#         return

#     # Check dataset
#     if not check_dataset():
#         print("⚠️ Please fix dataset issues before continuing.")
#         print(f"Expected dataset path: {DATASET_PATH}")
#         return

#     # Check if embeddings need to be created
#     try:
#         ingest_ipl_data()
#     except Exception as e:
#         print(f"⚠️ Error during data ingestion: {e}")
#         print("Proceeding with fallback mechanism only")

#     # Create RAG chain
#     try:
#         rag_chain = create_ipl_rag_chain()
#         if rag_chain:
#             print("✅ RAG chain initialized successfully")
#         else:
#             print("⚠️ RAG chain initialization failed")
#             print("Proceeding with fallback mechanism only")
#     except Exception as e:
#         print(f"⚠️ Error initializing RAG chain: {e}")
#         print("Proceeding with fallback mechanism only")
#         rag_chain = None

#     # Interactive loop
#     while True:
#         try:
#             # Get user question
#             question = input("\nAsk your IPL question (or type 'exit' to quit, 'update' to refresh embeddings): ")

#             if question.lower() in ['exit', 'quit', 'q']:
#                 print("Goodbye!")
#                 break
#             elif question.lower() == 'update':
#                 update_embeddings()
#                 # Reinitialize RAG chain with new embeddings
#                 try:
#                     rag_chain = create_ipl_rag_chain()
#                     if rag_chain:
#                         print("✅ RAG chain reinitialized successfully")
#                     else:
#                         print("⚠️ RAG chain reinitialization failed")
#                 except Exception as e:
#                     print(f"⚠️ Error reinitializing RAG chain: {e}")
#                     print("Proceeding with fallback mechanism only")
#                     rag_chain = None
#                 continue

#             print("\n🏏 IPL Analysis:\n")

#             try:
#                 # Try using the RAG chain if available
#                 if rag_chain:
#                     response = rag_chain.invoke({"input": question})
#                     answer = response.get("answer", "")

#                     # Check if we got a reasonable response
#                     if answer and len(answer) > 20:
#                         print(answer)
#                     else:
#                         # If response is too short or empty, use fallback
#                         print(fallback_response(question))
#                 else:
#                     # Use fallback if RAG chain not available
#                     print(fallback_response(question))

#             except Exception as e:
#                 print(f"Error generating response: {e}")
#                 print("\nFallback Analysis:\n")
#                 print(fallback_response(question))
#         except KeyboardInterrupt:
#             print("\nProgram interrupted. Exiting...")
#             break
#         except Exception as e:
#             print(f"Unexpected error: {e}")
#             print("Please try again.")

# if __name__ == "__main__":
#     main()
