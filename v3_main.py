import os
import time
from dotenv import load_dotenv
import pinecone
from pinecone import ServerlessSpec, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

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


def store_in_pinecone(chunks, embeddings, index_name=INDEX_NAME):
    """
    Store document chunks in Pinecone vector database.
    
    Args:
        chunks (list): List of document chunks
        embeddings (object): Embedding model
        index_name (str): Name of the Pinecone index
        
    Returns:
        object: Vector store for retrieval
    """
    print(f"Storing {len(chunks)} chunks in Pinecone index '{index_name}'...")
    
    vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=PINECONE_API_KEY  # Pass the API key explicitly
)
    
    print("Documents successfully stored in Pinecone")
    
    return vector_store


def perform_rag(vector_store, query, model_name="gpt-3.5-turbo"):
    """
    Perform Retrieval-Augmented Generation (RAG) on a query.
    
    Args:
        vector_store (object): Vector store containing document embeddings
        query (str): User query to process
        model_name (str): Name of the LLM to use for generation
        
    Returns:
        str: Generated response to the query
    """
    print(f"Performing RAG for query: '{query}'")
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create a language model
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Create a RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    # Run the chain
    response = rag_chain.run(query)
    
    return response


def main(pdf_directory, query=None):
    """
    Main function to run the RAG pipeline.
    
    Args:
        pdf_directory (str): Path to the directory containing PDF files
        query (str, optional): Query to process
        
    Returns:
        object: Vector store and RAG response if query provided
    """
    # Load and process documents
    documents = load_pdfs(pdf_directory)
    chunks = chunk_text(documents)
    
    # Initialize embedding model and Pinecone
    embeddings = generate_embeddings()
    _, _ = create_pinecone_index_if_not_exists()
    
    # Store documents in Pinecone
    vector_store = store_in_pinecone(chunks, embeddings)
    
    # Perform RAG if query provided
    if query:
        response = perform_rag(vector_store, query)
        return vector_store, response
    
    return vector_store


# Example usage
if __name__ == "__main__":
    DATA_DIR_PATH = "pdfs/Endowment_plans"
    
    # Option 1: Just create the vector store
    vector_store = main(DATA_DIR_PATH)
    
    # Option 2: Create vector store and perform a query
    # sample_query = "What are the key features of these endowment plans?"
    # vector_store, response = main(DATA_DIR_PATH, sample_query)
    # print(f"Response: {response}")