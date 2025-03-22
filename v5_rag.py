import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configuration constants
PINECONE_API_KEY = os.environ['PINE_KEY']
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
INDEX_NAME = "embed-2"
EMBEDDING_MODEL = "text-embedding-3-small"
PINECONE_HOST = "https://embed-2-6kgtl49.svc.aped-4627-b74a.pinecone.io"


def setup_resources(index_name=INDEX_NAME):
    """Initialize Pinecone index and embeddings model"""
    print(f"Setting up resources for index '{index_name}'...")
    
    # Initialize Pinecone and embeddings
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name, host=PINECONE_HOST)
    embeddings_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Log index stats
    stats = index.describe_index_stats()
    print(f"Connected to index with {stats['total_vector_count']} vectors")
    
    return index, embeddings_model


def retrieve_documents(index, query, embeddings_model, top_k=5, namespace=None, filter=None):
    """Search for relevant documents in Pinecone"""
    print(f"Searching for: '{query}'")
    
    # Create query embedding and search parameters
    query_embedding = embeddings_model.embed_query(query)
    search_params = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
        **({"namespace": namespace} if namespace else {}),
        **({"filter": filter} if filter else {})
    }
    
    # Search and process results
    results = index.query(**search_params)
    documents = []
    
    for match in results['matches']:
        # Extract text and metadata
        text = match['metadata'].pop('text', "")
        metadata = {**match['metadata'], 'score': match['score']}
        documents.append(Document(page_content=text, metadata=metadata))
    
    print(f"Found {len(documents)} relevant documents")
    return documents


def rag_query(query, index=None, embeddings_model=None, model_name="gpt-4o", 
             temperature=0.7, top_k=5, index_name=INDEX_NAME, max_tokens=None):
    """Perform RAG query and generate response"""
    # Initialize resources if not provided
    if not index or not embeddings_model:
        index, embeddings_model = setup_resources(index_name)
    
    # Retrieve documents
    docs = retrieve_documents(
        index=index,
        query=query,
        embeddings_model=embeddings_model,
        top_k=top_k
    )
    
    # Handle case with no results
    if not docs:
        return {
            'query': query,
            'response': "I couldn't find any relevant information to answer your question.",
            'source_documents': [],
            'metadata': {'success': False, 'reason': 'No relevant documents found'}
        }
    
    # Build context from documents
    context_parts = [
        f"Document {i+1} [Source: {doc.metadata.get('filename', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 'Unknown')}]:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    context = "\n\n".join(context_parts)
    print(docs)
    # Create prompt and generate response
    prompt = f"""
    Answer the following question based ONLY on the provided context.
    If you cannot answer the question based on the context, say "I don't have enough information to answer this question."
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=max_tokens
    )
    
    response = llm.invoke(prompt)
    
    # Return results
    return {
        'query': query,
        'response': response.content,
        'source_documents': docs,
        'metadata': {
            'model': model_name,
            'temperature': temperature,
            'top_k': top_k,
            'num_docs_retrieved': len(docs)
        }
    }


def interactive_query():
    """Interactive CLI for querying the system"""
    index, embeddings_model = setup_resources()
    
    print("\n===== RAG Query System =====")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        
        result = rag_query(
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
    interactive_query()