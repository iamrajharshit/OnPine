import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configuration constants
PINECONE_API_KEY = os.environ['PINE_KEY']
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
INDEX_NAME = "embeds"
EMBEDDING_MODEL = "text-embedding-3-small"


def initialize_resources(index_name=INDEX_NAME):
    """
    Initialize the necessary resources for querying.
    
    Args:
        index_name (str): Name of the Pinecone index to query
        
    Returns:
        tuple: (pinecone_index, embeddings_model)
    """
    print(f"Initializing resources for querying index '{index_name}'...")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name,host="https://embeds-6kgtl49.svc.aped-4627-b74a.pinecone.io")
    
    # Initialize embedding model
    embeddings_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"Connected to index '{index_name}' with {stats['total_vector_count']} vectors")
    
    return index, embeddings_model


def search_documents(index, query_text, embeddings_model, top_k=5, namespace=None, filter=None):
    """
    Search for documents in the Pinecone index.
    
    Args:
        index: Pinecone index object
        query_text (str): The query text to search for
        embeddings_model: Model to generate embeddings
        top_k (int): Number of results to return
        namespace (str, optional): Namespace to search in
        filter (dict, optional): Metadata filters
        
    Returns:
        list: List of Document objects with content and metadata
    """
    print(f"Searching for: '{query_text}'")
    
    # Generate embedding for the query
    query_embedding = embeddings_model.embed_query(query_text)
    
    # Search parameters
    search_params = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True
    }
    
    # Add optional parameters if provided
    if namespace:
        search_params["namespace"] = namespace
    if filter:
        search_params["filter"] = filter
    
    # Perform search
    results = index.query(**search_params)
    
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
    
    print(f"Found {len(documents)} relevant documents")
    
    return documents

#gpt-3.5-turbo
def perform_rag_query(query, index=None, embeddings_model=None, model_name="gpt-4o", 
                     temperature=0.7, top_k=5, index_name=INDEX_NAME, max_tokens=None):
    """
    Perform a RAG query against the Pinecone index.
    
    Args:
        query (str): The query text
        index (object, optional): Existing Pinecone index object
        embeddings_model (object, optional): Existing embeddings model
        model_name (str): LLM model to use
        temperature (float): LLM temperature
        top_k (int): Number of documents to retrieve
        index_name (str): Pinecone index name (used if index not provided)
        max_tokens (int, optional): Max tokens for response
        
    Returns:
        dict: {
            'query': original query,
            'response': generated response,
            'source_documents': list of source documents,
            'metadata': metadata about the query
        }
    """
    # Initialize resources if not provided
    if index is None or embeddings_model is None:
        index, embeddings_model = initialize_resources(index_name)
    
    # Search for relevant documents
    retrieved_docs = search_documents(
        index=index,
        query_text=query,
        embeddings_model=embeddings_model,
        top_k=top_k
    )
    
    if not retrieved_docs:
        return {
            'query': query,
            'response': "I couldn't find any relevant information to answer your question.",
            'source_documents': [],
            'metadata': {'success': False, 'reason': 'No relevant documents found'}
        }
    
    # Create context from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        # Add document with source
        source = doc.metadata.get('source', 'Unknown source')
        filename = doc.metadata.get('filename', 'Unknown file')
        page = doc.metadata.get('page', 'Unknown page')
        
        context_parts.append(f"Document {i+1} [Source: {filename}, Page: {page}]:\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    # Create the prompt
    prompt = f"""
    Answer the following question based ONLY on the provided context.
    If you cannot answer the question based on the context, say "I don't have enough information to answer this question."
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    # Create language model
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=max_tokens
    )
    
    # Generate response
    response = llm.invoke(prompt)
    
    # Prepare result with metadata
    result = {
        'query': query,
        'response': response.content,
        'source_documents': retrieved_docs,
        'metadata': {
            'model': model_name,
            'temperature': temperature,
            'top_k': top_k,
            'num_docs_retrieved': len(retrieved_docs)
        }
    }
    
    return result


def filter_by_metadata(index, embeddings_model, query_text, filter_dict, top_k=5):
    """
    Search documents with metadata filtering.
    
    Args:
        index: Pinecone index
        embeddings_model: Embeddings model
        query_text (str): Query text
        filter_dict (dict): Dictionary of metadata filters
        top_k (int): Number of results to return
        
    Returns:
        list: List of Document objects with content and metadata
    """
    print(f"Searching for: '{query_text}' with filters: {filter_dict}")
    
    # Use the search_documents function with filters
    documents = search_documents(
        index=index,
        query_text=query_text,
        embeddings_model=embeddings_model,
        top_k=top_k,
        filter=filter_dict
    )
    
    return documents


# Sample usage functions

def query_index():
    """Interactive command-line function to query the index."""
    index, embeddings_model = initialize_resources()
    
    print("\n===== RAG Query System =====")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        
        result = perform_rag_query(
            query=query,
            index=index,
            embeddings_model=embeddings_model,
            top_k=10
        )
        
        print("\n" + "="*50)
        print("RESPONSE:")
        print(result['response'])
        print("="*50)
        print(f"Retrieved {len(result['source_documents'])} documents")


if __name__ == "__main__":
    # Example usage:
    query_index()
    
    # Alternative: Single query
    # query = "What are the key benefits of endowment plans?"
    # result = perform_rag_query(query)
    # print(result['response'])