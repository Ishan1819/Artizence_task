# import pandas as pd
# import chromadb
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# import google.generativeai as genai

# # Step 1: Load CSV
# df = pd.read_csv("D:/Ishan_ip datasets/player_metrics.csv")
# df=df.tail(3000)
# texts = df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1).tolist()

# # Step 2: Use HuggingFace model identifier for embeddings
# embedding_model_name = "paraphrase-MiniLM-L6-v2"
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# # Step 3: Set up ChromaDB to store embeddings
# client = chromadb.Client()
# collection = client.create_collection(name="my_collection")

# # Insert documents into Chroma with non-empty metadata
# documents = []
# metadatas = []
# ids = []

# for i, text in enumerate(texts):
#     documents.append(text)
#     # Add non-empty metadata for each document (required by ChromaDB)
#     metadatas.append({"source": "player_metrics.csv", "index": i})
#     ids.append(str(hash(text)))

# # Add all documents at once
# collection.add(
#     documents=documents,
#     metadatas=metadatas,
#     ids=ids
# )

# # Step 4: Set up Gemini API
# genai.configure(api_key="AIzaSyDEuV85vMTs040I3S5U6ZbyVH4dNEtv9KA")
# model = genai.GenerativeModel("gemini-1.5-flash")

# # Step 5: Define the RAG pipeline with ChromaDB and Gemini
# def ask_rag(question, top_k=3):
#     # Get the query embedding
#     query_embedding = embeddings.embed_query(question)  # Use embed_query for a single query
    
#     # Query the collection
#     results = collection.query(
#         query_embeddings=[query_embedding],  # Needs to be a list
#         n_results=top_k
#     )
    
#     # Process the results
#     retrieved_texts = results['documents'][0]  # First element contains the list of documents
    
#     # Combine the retrieved documents into context
#     context = "\n\n".join(retrieved_texts)
    
#     # Generate a prompt for the Gemini model
#     prompt = f"""Answer the question based on the following data:

# {context}

# Question: {question}"""
    
#     # Get the response from Gemini
#     response = model.generate_content(prompt)
#     return response.text

# # Example usage
# # if __name__ == "__main__":
# query = "How much will V Kohli score in this match?"
# print(ask_rag(query))



import google.generativeai as genai

# Configure the Gemini API with your key
genai.configure(api_key="AIzaSyDEuV85vMTs040I3S5U6ZbyVH4dNEtv9KA")

# Create a model instance
model = genai.GenerativeModel("gemini-1.5-flash")

def ask_gemini(question):
    """Send a question directly to Gemini with IPL analyst prompt"""
    # Format the prompt properly for the Gemini API
    prompt = f"""You are an IPL analyst expert and you can predict the upcoming match player scores, their wickets taking and other factors from the last 4 to 5 matches they played. Calculate this data from last 4 to 5 matches and give me the response.

Your responses should be concise and follow this format:
- For batting questions: Analyze performance from last 5 matches and predict runs with strike rate
- For bowling questions: Analyze performance from last 5 matches and predict wickets with economy rate
- Include relevant stats that support your prediction

Example:
Q: How much will Virat Kohli score today vs MI?
A: According to the last 5 matches he played, he would score 50+ runs with a strike rate of 120.

Question: {question}"""
    
    # Generation configuration
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
    }
    
    # Get the response from Gemini using a single prompt
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    
    return response.text

# Example usage
# if __name__ == "__main__":
query = "How much will V Kohli score in this match against MI?"
answer = ask_gemini(query)
print("\nIPL Analyst Prediction:")
print(answer)