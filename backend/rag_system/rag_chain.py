from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import VECTOR_STORE_PATH


def create_ipl_rag_chain():
    model = ChatOllama(model="iplguru")

    # Engineering the prompt to guide the model in generating a structured, detailed explanation
    prompt = PromptTemplate.from_template(
        """
        <s> [INST] <<SYS>>
        You are an expert IPL cricket analyst chatbot. Based on your expertise and historical match data, you need to provide an explanation for the following IPL match prediction.
        <</SYS>>
        Context: {context}
        
        The user has predicted the following winner:
        Predicted Winner: {input}

        Please provide a concise and very short points with 20 to 25 words only each point, step-by-step explanation for this prediction, covering the following points:
        - Historical performance of the two teams
        - Recent match results and any relevant trends
        - Team strengths and weaknesses based on the current form
        - Any other relevant factors like player performance, pitch conditions, or head-to-head statistics.

        Your explanation should justify the prediction made, based on the data and analysis provided. Please give a detailed analysis in a structured manner.
        [/INST]
        """
    )

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH, embedding_function=embedding
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5},
    )

    document_chain = create_stuff_documents_chain(model, prompt)

    rag_chain = {"context": retriever, "input": RunnablePassthrough()} | document_chain

    return rag_chain


rag_chain = create_ipl_rag_chain()


def run_rag(prompt: str) -> str:
    response = rag_chain.invoke(prompt)
    return response