import os
from typing import List, Dict, Any
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import uuid
import numpy as np
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


# Set your API keys
GOOGLE_API_KEY = os.environ['GO_KEY']
PINECONE_API_KEY = os.environ['PINE_KEY']

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Configuration
INDEX_NAME = "pdf-embeddings"
VECTOR_DIMENSION = 768  # Dimension for text-embedding-gecko
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200



def perform_rag(query: str, top_k: int = 40) -> str: #top_k 5,10,40
    """Perform RAG by retrieving relevant chunks and generating a response with Gemini."""
    # Generate embedding for the query
    query_result = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = query_result["embedding"]
    
    # Query Pinecone with the specific host provided
    index = pc.Index(host="https://pdf-embeddings-6kgtl49.svc.aped-4627-b74a.pinecone.io")
    query_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract relevant context from the results
    context_texts = []
    for match in query_results["matches"]:
        context_texts.append(f"Source: {match['metadata']['source']}\n{match['metadata']['text']}")
    
    context = "\n\n".join(context_texts)
    
    # Prepare prompt for Gemini
    prompt = f"""
    Based on the following context, please answer the question:
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    
    ANSWER:
    """
    
    # Generate response with Gemini
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    response = model.generate_content(prompt)
    
    return response.text


def main():

    # Example of querying the system
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        print("\nRetrieving information and generating response...")
        response = perform_rag(query)
        print("\nAnswer:")
        print(response)

if __name__ == "__main__":
    main()