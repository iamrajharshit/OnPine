import os
import time  # Import time for sleep functionality
from dotenv import load_dotenv
import pinecone
from pinecone import ServerlessSpec, Pinecone  # Import necessary Pinecone classes
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split the text into smaller chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For Google Generative AI embeddings
from langchain_openai import OpenAIEmbeddings # To create embeddings
from langchain_pinecone import PineconeVectorStore  # To connect with the Pinecone Vectorstore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader  # To load and parse PDFs

# Load environment variables from .env file
load_dotenv()

# Set your API keys from environment variables
GOOGLE_API_KEY = os.environ['GO_KEY']
PINECONE_API_KEY = os.environ['PINE_KEY']
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

# Define the name for the Pinecone index
index_name = "embeds"  # Name for the index; can use an existing index or create a new one

# Initialize Pinecone client with the API key
pc = Pinecone(api_key=PINECONE_API_KEY)

# Print the Pinecone client to confirm successful initialization
print(pc)

# Check if the index already exists
if index_name in pc.list_indexes().names():
    print("Index already exists:", index_name)
    index = pc.Index(index_name)  # Use the existing index
    print(index.describe_index_stats())  # Print index statistics
else:
    # Create a new index with specified parameters
    pc.create_index(
        name=index_name,
        dimension=1536,  # Dimensions
        metric="cosine",  # Metric
        spec=ServerlessSpec(
            cloud="aws",  # Cloud
            region="us-east-1",  # Region
            type="Dense",  # Type
            capacity_mode="Serverless"  # Capacity mode
        ),
        embedding_model="text-embedding-3-small"  # Embedding model
    )
    # Wait until the index is ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)  # Sleep for a short duration before checking again
    index = pc.Index(index_name)  # Initialize the new index
    print("Index created")
    print(index.describe_index_stats())  # Print index statistics for the new index




DATA_DIR_PATH = "pdfs/Endowment_plans" # Directory containing our PDF files
CHUNK_SIZE = 1024 # Size of each text chunk for processing
CHUNK_OVERLAP = 100 # Amount of overlap between chunks
INDEX_NAME = index_name # Name of our Pinecone index


from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
loader = DirectoryLoader(
    path=DATA_DIR_PATH,  # Directory containing our PDFs
    glob="**/*.pdf",     # Pattern to match PDF files (including subdirectories)
    loader_cls=PyPDFLoader  # Specifies we're loading PDF files
)
docs = loader.load()  # This loads all matching PDF files
print(f"Total Documents loaded: {len(docs)}")


# we can convert the Document object to a python dict using the .dict() method.
print(f"keys associated with a Document: {docs[0].dict().keys()}")


print(f"{'-'*15}\nFirst 100 charachters of the page content: {docs[0].page_content[:100]}\n{'-'*15}")
print(f"Metadata associated with the document: {docs[0].metadata}\n{'-'*15}")
print(f"Datatype of the document: {docs[0].type}\n{'-'*15}")

#  We loop through each document and add additional metadata - filename, quarter, and year
for doc in docs:
    filename = doc.dict()['metadata']['source'].split("\\")[-1]
    #quarter = doc.dict()['metadata']['source'].split("\\")[-2]
    #year = doc.dict()['metadata']['source'].split("\\")[-3]
    doc.metadata = {"filename": filename, "source": doc.dict()['metadata']['source'], "page": doc.dict()['metadata']['page']}

# To veryfy that the metadata is indeed added to the document
print(f"Metadata associated with the document: {docs[0].metadata}\n{'-'*15}")
print(f"Metadata associated with the document: {docs[1].metadata}\n{'-'*15}")
print(f"Metadata associated with the document: {docs[2].metadata}\n{'-'*15}")
print(f"Metadata associated with the document: {docs[3].metadata}\n{'-'*15}")


for i in range(len(docs)) :
  print(f"Metadata associated with the document: {docs[i].metadata}\n{'-'*15}")



from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1024,
chunk_overlap=100
)
documents = text_splitter.split_documents(docs)
print(documents)


# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
documents = text_splitter.split_documents(docs)
print(len(docs), len(documents))

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small") # Initialize the embedding model
print(embeddings)

#Step7: Embedding and Vector Store Creation

# Using the Vector Store for Retrieval
# Here we will define how to use the loaded vectorstore as retriver


def load_pdfs():
    return 



def chunk_text():
    return

def chunk_text():
    return

