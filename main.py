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

def load_pdfs(pdf_directory: str) -> Dict[str, str]:
    """Load and extract text from all PDFs in the given directory."""
    pdf_texts = {}
    
    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(pdf_directory, filename)
            
            try:
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text()
                
                pdf_texts[filename] = text
                print(f"Successfully extracted text from {filename}")
            except Exception as e:
                print(f"Error extracting text from {filename}: {e}")
    
    return pdf_texts

def chunk_text(pdf_texts: Dict[str, str]) -> List[Dict[str, Any]]:
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    all_chunks = []
    
    for filename, text in pdf_texts.items():
        chunks = text_splitter.split_text(text)
        
        for chunk in chunks:
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {
                    "source": filename
                }
            })
    
    print(f"Created {len(all_chunks)} chunks from {len(pdf_texts)} PDFs")
    return all_chunks

def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for each chunk using Gemini."""
    embedding_model = "models/embedding-001"
    
    for chunk in chunks:
        try:
            result = genai.embed_content(
                model=embedding_model,
                content=chunk["text"],
                task_type="retrieval_document"
            )
            
            # Get the embedding values
            chunk["embedding"] = result["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            chunk["embedding"] = None
    
    # Filter out any chunks that failed to get embeddings
    chunks = [chunk for chunk in chunks if chunk["embedding"] is not None]
    
    print(f"Generated embeddings for {len(chunks)} chunks")
    return chunks

def create_pinecone_index_if_not_exists():
    """Create Pinecone index if it doesn't exist already or use existing one."""
    if INDEX_NAME not in pc.list_indexes().names():
        try:
            # Create with the exact configuration provided
            pc.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created new Pinecone index: {INDEX_NAME} on AWS us-east-1")
        except Exception as e:
            print(f"Error creating index: {e}")
            print("The index might already exist with a different configuration.")
            print("Using the existing index if available...")
    else:
        print(f"Using existing Pinecone index: {INDEX_NAME}")

def store_in_pinecone(chunks_with_embeddings: List[Dict[str, Any]]):
    """Store chunks and their embeddings in Pinecone."""
    create_pinecone_index_if_not_exists()
    
    # Use the index with the specific host provided
    index = pc.Index(host="https://pdf-embeddings-6kgtl49.svc.aped-4627-b74a.pinecone.io")
    
    # Prepare the vectors for upserting
    vectors_to_upsert = []
    for chunk in chunks_with_embeddings:
        vectors_to_upsert.append({
            "id": chunk["id"],
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["text"],
                "source": chunk["metadata"]["source"]
            }
        })
    
    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        index.upsert(vectors=batch)
    
    print(f"Stored {len(vectors_to_upsert)} vectors in Pinecone")

def perform_rag(query: str, top_k: int = 5) -> str:
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
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    
    return response.text

def main():
    print("Please select the Listed Policy By LIC:")
    print("1. lic_india_whole_life_plan")
    print("2. money_back_plans")
    print("3. terms_assurance_plans")
    print("4. riders")
    print("5. Endowment_plans")
    
    choice = input("Enter the number corresponding to your choice: ")
    
    if choice == '1':
        pdf_directory = "lic_india_whole_life_plan"
    elif choice == '2':
        pdf_directory = "money_back_plans"

    elif choice == '3':
        pdf_directory="terms_assurance_plans"

    elif choice == '4':
        pdf_directory="riders"

    elif choice=="5":
        pdf_directory="Endowment_plans"
    else:
        print("Invalid choice. Defaulting to 'lic_india_whole_life_plan'.")
        pdf_directory = "lic_india_whole_life_plan"
    



    # Process the PDFs
    print("Step 1: Loading PDFs...")
    pdf_texts = load_pdfs(pdf_directory)
    
    print("\nStep 2: Chunking text...")
    chunks = chunk_text(pdf_texts)
    
    print("\nStep 3: Generating embeddings...")
    chunks_with_embeddings = generate_embeddings(chunks)
    
    print("\nStep 4: Storing in Pinecone...")
    store_in_pinecone(chunks_with_embeddings)
    
    print("\nRAG system is ready! You can now query your documents.")
    
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