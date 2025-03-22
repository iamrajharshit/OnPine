import os
import time
import uuid
from dotenv import load_dotenv
import pinecone
from pinecone import ServerlessSpec, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.schema import Document
import numpy as np

# Load environment variables
load_dotenv()

# Configuration constants
PINECONE_API_KEY = os.environ['PINE_KEY']
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
INDEX_NAME = "embeds"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def load_pdfs(directory_path):
    """
    Load PDF documents from a specified directory.
    
    Args:
        directory_path (str): Path to the directory containing PDF files
        
    Returns:
        list: List of loaded document objects
    """
    print(f"Loading PDFs from {directory_path}...")
    
    loader = DirectoryLoader(
        path=directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    
    docs = loader.load()
    print(f"Total documents loaded: {len(docs)}")
    
    # Enhance metadata for each document
    for doc in docs:
        filename = doc.metadata['source'].split('\\')[-1]
        doc.metadata = {
            "filename": filename, 
            "source": doc.metadata['source'], 
            "page": doc.metadata['page']
        }
    
    return docs


def chunk_text(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents (list): List of document objects
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Amount of overlap between chunks
        
    Returns:
        list: List of chunked document objects
    """
    print(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Original documents: {len(documents)}, Chunks created: {len(chunks)}")
    
    return chunks


def generate_embeddings(model=EMBEDDING_MODEL):
    """
    Initialize the embedding model.
    
    Args:
        model (str): Name of the embedding model to use
        
    Returns:
        object: Initialized embedding model
    """
    print(f"Initializing embedding model: {model}")
    
    embeddings = OpenAIEmbeddings(
        model=model,
        openai_api_key=OPENAI_API_KEY
    )
    
    return embeddings


def create_pinecone_index_if_not_exists(index_name=INDEX_NAME, dimension=EMBEDDING_DIMENSION):
    """
    Check if Pinecone index exists, create if it doesn't.
    
    Args:
        index_name (str): Name of the Pinecone index
        dimension (int): Dimension of the embeddings
        
    Returns:
        object: Initialized Pinecone index
    """
    print("Connecting to Pinecone...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if index_name in pc.list_indexes().names():
        print(f"Index already exists: {index_name}")
        index = pc.Index(index_name)
        print(index.describe_index_stats())
    else:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
                type="Dense",
                capacity_mode="Serverless"
            ),
            embedding_model=EMBEDDING_MODEL
        )
        
        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            print("Waiting for index to be ready...")
            time.sleep(1)
            
        index = pc.Index(index_name)
        print("Index created successfully")
        print(index.describe_index_stats())
    
    return pc, index


def store_in_pinecone(chunks, embeddings_model, index_name=INDEX_NAME):
    """
    Store document chunks in Pinecone vector database using direct API.
    
    Args:
        chunks (list): List of document chunks
        embeddings_model (object): Embedding model
        index_name (str): Name of the Pinecone index
        
    Returns:
        object: Pinecone index for retrieval
    """
    print(f"Storing {len(chunks)} chunks in Pinecone index '{index_name}'...")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name,host="https://embeds-6kgtl49.svc.aped-4627-b74a.pinecone.io")
    
    # Process chunks in batches to avoid timeouts
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        # Prepare vectors for batch insertion
        vectors_to_upsert = []
        
        for chunk in batch:
            # Generate embedding for the text content
            embedding = embeddings_model.embed_query(chunk.page_content)
            
            # Create a unique ID for this vector
            vector_id = str(uuid.uuid4())
            
            # Create vector record with metadata
            vector = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.page_content,
                    **chunk.metadata  # Include original metadata
                }
            }
            
            vectors_to_upsert.append(vector)
        
        # Upsert vectors to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        
        print(f"Batch {i//batch_size + 1} uploaded: {len(vectors_to_upsert)} vectors")
    
    # Get updated stats
    stats = index.describe_index_stats()
    print(f"Upload complete. Total vectors in index: {stats['total_vector_count']}")
    
    return index


def perform_search(index, query_text, embeddings_model, top_k=5):
    """
    Search Pinecone index using a query string.
    
    Args:
        index: Pinecone index
        query_text (str): Query text
        embeddings_model: Model to generate query embedding
        top_k (int): Number of results to return
        
    Returns:
        list: List of Document objects with relevant text and metadata
    """
    # Generate embedding for the query
    query_embedding = embeddings_model.embed_query(query_text)
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host="https://embeds-6kgtl49.svc.aped-4627-b74a.pinecone.io")

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Convert results to Document objects
    documents = []
    for match in results['matches']:
        # Extract the text and metadata
        text = match['metadata'].pop('text', "")
        metadata = match['metadata']
        
        # Add score to metadata
        metadata['score'] = match['score']
        
        # Create Document object
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
    
    return documents


def perform_rag(index, query, embeddings_model, model_name="gpt-3.5-turbo"):
    """
    Perform Retrieval-Augmented Generation (RAG) on a query.
    
    Args:
        index: Pinecone index
        query (str): User query to process
        embeddings_model: Model to generate query embedding
        model_name (str): Name of the LLM to use for generation
        
    Returns:
        str: Generated response to the query
    """
    print(f"Performing RAG for query: '{query}'")
    
    # Retrieve relevant documents
    docs = perform_search(index, query, embeddings_model)
    
    if not docs:
        return "No relevant information found."
    
    # Create context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create the prompt
    prompt = f"""
    Answer the following question based on the provided context.
    If you cannot answer the question based on the context, say "I don't have enough information to answer this question."
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    # Create a language model
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Generate response
    response = llm.invoke(prompt)
    
    return response.content


def main(pdf_directory, query=None):
    """
    Main function to run the RAG pipeline.
    
    Args:
        pdf_directory (str): Path to the directory containing PDF files
        query (str, optional): Query to process
        
    Returns:
        object: Pinecone index and RAG response if query provided
    """
    # Load and process documents
    documents = load_pdfs(pdf_directory)
    chunks = chunk_text(documents)
    
    # Initialize embedding model and Pinecone
    embeddings_model = generate_embeddings()
    _, _ = create_pinecone_index_if_not_exists()
    
    # Store documents in Pinecone
    index = store_in_pinecone(chunks, embeddings_model)
    
    # Perform RAG if query provided
    if query:
        response = perform_rag(index, query, embeddings_model)
        return index, response
    
    return index


# Example usage
if __name__ == "__main__":
    DATA_DIR_PATH = "pdfs/Endowment_plans"
    
    # Option 1: Just create the vector store
    index = main(DATA_DIR_PATH)
    
    # # Option 2: Create vector store and perform a query
    # sample_query = "What are the key features of these endowment plans?"
    # index, response = main(DATA_DIR_PATH, sample_query)
    # print(f"Response: {response}")