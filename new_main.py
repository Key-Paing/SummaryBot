from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import VertexAI
from langchain_experimental.text_splitter import SemanticChunker

import psycopg2

# DB connection
conn = psycopg2.connect(
    host="localhost",
    database="",
    user="",
    password=""
)
cursor = conn.cursor()

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

session_state = {}

#Fetching Cases from the database by case ID
def fetch_case_by_id(case_id):
    cursor.execute("SELECT * FROM cases WHERE id = %s", (case_id,))
    case = cursor.fetchone()
    if case:
        return {
            "id": case[0],
            "title": case[1],
            "content": case[2]
        }
    return None


# Function to embedding the case content and retrieve chunks[cases]
def get_or_create_retriever(case_id, text):
    if case_id in session_state:
        return session_state[case_id]

    case = fetch_case_by_id(case_id)
    if not case:
        raise ValueError(f"Case with ID {case_id} not found.")

    # Create a retriever for the case content
    chunker = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")
    semantic_chunks = chunker.create_documents([text])
    vector_store = FAISS.from_documents(semantic_chunks, embedding_model)
    semantic_chunk_retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    session_state[f"retriever_{case_id}"] = semantic_chunk_retriever
    session_state[f"semantic_chunks_{case_id}"] = semantic_chunks
    session_state[f"embedding_done_{case_id}"] = True

    return semantic_chunk_retriever


# Function for LLM to summarize based on the case content
def load_llm():
    return VertexAI(
                project = "machine-translation-001",
                location = "us-central1",
                model = "gemini-2.5-pro-preview-05-06",
                credentials=credentials, 
    )

# Function to summarize the case content by ID
def summarize_cases_by_id(case_ids):

    llm = load_llm()

    prompt = ChatPromptTemplate.from_template(

        """ You are a legal analyst specializing in Burmese law. 
            Your task is to analyze the following Burmese legal content and generate a short summary in Burmese.
            The output can be written in casual language, but must be organized into the following sections:
                                - Case summary of the case.
                                - Court Findings.
                                - Judgement of the court.

              User's Query:
                {question}

              Burmese Legal Content:
                {context}"""
    )

    results = []

    for case_id in case_ids:
        case = fetch_case_by_id(case_id)
        if not case:
            continue

    # semantic_chunk_retriever = session_state[f"retriever_{case_id}", case["content"]]
    semantic_chunk_retriever = get_or_create_retriever(case_id, case["content"])

    if session_state.get(f"summary_done_{case_id}"):
        summary = session_state[f"summary_{case_id}"]

    else:
        rag_chain = (
        {"context": semantic_chunk_retriever, "question": RunnablePassthrough() }
         | prompt
         | llm
         | StrOutputParser()
        )

        summary = rag_chain.invoke("Summarize")
        session_state[f"summary_{case_id}"] = summary
        session_state[f"summary_done_{case_id}"] = True

    

    results.append({
        "id": case["id"],
        "title": case["title"],
        "summary": summary
    })

    return results

