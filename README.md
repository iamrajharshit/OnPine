# Understanding Pinecone Vector Databases
## WHat is Vector Database?
Vector databases are specialized storage systems optimized for managing high-dimensional vector data.  
Unlike traditional relational databases that use row-column structures, vector databases employ advanced  
indexing algorithms to organize and query numerical vector representations of data points in n-dimensional space.

### Need for Vector DB
Large Language Models process and genrate text based on vast amounts of training data.
Vector databses enhance LLM capabilities by:
- Semantic Search
- Retrival Augmented Generation (RAG)
- Scalable Information Retrival
- Low-latency Querying
  
### Features of Pinecone Vector Database
- Indexing Algorithms
    - Hierarchical Navigable Small World (HNSW) graphs for efficient ANN search.
    - Optimized for high recall and low latency in high-dimensional spaces.
- Scalability
    - Distributed architecture supporting billions of vectors.
    - Automatic sharding and load balancing for horizontal scaling.
- Real-time Operations
    - Support for concurrent reads and writes.
    - Immediate consistency for index updates.
- Query Capabilities
    - Metadata filtering for hybrid searches.
    - Support for batched queries to optimize throughput.
- Vector Optimizations
    - Quantization techniques to reduce memory footprint.
    - Efficient compression methods for vector storage.
## Getting Started with Pinecone
Here, we will discuss Pinecone Index, ingesting Data that, PDF files and will develop a retriever.
Index repesents the hightest level organizational unit of vector data.
- Pinecone's core data units, vectors, are accepted and stored using an index.
- It serves queries over the vectors it contains, allowing you to search for similar vectors.
- Think of an index as a specialized database for vector data. When you make an index, you provide essential characteristics.
- The vectors’ dimension (such as 2-dimensional, 768-dimensional, etc.) that needs to be stored 2.
- The query-specific similarity measure (e.g., cosine similarity, Euclidean etc.)
- Also we can chose the dimension as per model like if we choose mistral embed model then there will be 1024dimensions.
- Here we are using Gemini 1.5 pro
### Types of Indexes
- Serverless Indexes:These automatically scale based on usage, and you pay only for the amount of data stored and operations performed.
- Pod-based Indexes:These use pre-configured units of hardware (pods) that you choose based on your storage and performance needs.
  Understanding indexes is crucial because they form the foundation of how you organize and interact with your vector data in Pinecone.
  
