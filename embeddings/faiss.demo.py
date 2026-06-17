from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ----------------------------
# Step 1: Sample documents
# ----------------------------
documents = [
    "FastAPI generates API documentation",
    "Python supports object oriented programming",
    "Football is a popular sport",
    "LangChain helps build LLM applications",
    "FAISS performs vector similarity search"
]

# ----------------------------
# Step 2: Load embedding model
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Step 3: Create embeddings for documents
# ----------------------------
embeddings = model.encode(documents)

# print("embedding for documents -----", embeddings)

# embedding for documents ----- [[-0.08826308  0.06214033 -0.05554025 ...  0.10326625  0.09444451
#    0.09263745]
#  [-0.08313452  0.0200266  -0.03137647 ...  0.04651546  0.17490469
#    0.03420191]
#  [ 0.06504587  0.01154606 -0.01104547 ...  0.02196179  0.11883238
#    0.03942854]
#  [-0.01413894 -0.02463527  0.05002333 ... -0.03460881  0.05255089
#    0.01929881]
#  [-0.05954954 -0.04006417 -0.02244069 ... -0.04018053  0.05757446
#    0.0032613 ]]

# FAISS expects float32
embeddings = np.array(embeddings, dtype="float32")

# print("embedding  for documents in float32 -----", embeddings)

# embedding  for documents in float32 ----- [[-0.08826308  0.06214033 -0.05554025 ...  0.10326625  0.09444451
#    0.09263745]
#  [-0.08313452  0.0200266  -0.03137647 ...  0.04651546  0.17490469
#    0.03420191]
#  [ 0.06504587  0.01154606 -0.01104547 ...  0.02196179  0.11883238
#    0.03942854]
#  [-0.01413894 -0.02463527  0.05002333 ... -0.03460881  0.05255089
#    0.01929881]
#  [-0.05954954 -0.04006417 -0.02244069 ... -0.04018053  0.05757446
#    0.0032613 ]]

# ----------------------------
# Step 4: Create FAISS index
# ----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# print("Prining dimension for embedding---------------",dimension)
# Prining dimension for embedding--------------- 384

# print ("Dimension with index------", index)
# Dimension with index------ <faiss.swigfaiss.IndexFlatL2; proxy of <Swig Object of type 'faiss::IndexFlatL2 *' at 0x0000024FEF7DDAF0> >

# ----------------------------
# Step 5: Add document embeddings to index
# ----------------------------
index.add(embeddings)
# print(f"Total vectors stored in FAISS: {index.ntotal}")
# Total vectors stored in FAISS: 5

# ----------------------------
# Step 6: Define queries
# ----------------------------
queries = [
    "How does FastAPI create docs?",
    "What is vector search?",
    "Tell me about LangChain"
]

# ----------------------------
# Step 7: Search for each query
# ----------------------------
for query in queries:
    print("\n" + "=" * 80)
    # print(f"Query: {query}")
        #     ================================================================================
        # Query: How does FastAPI create docs?

        # ================================================================================
        # Query: What is vector search?

        # ================================================================================
        # Query: Tell me about LangChain

    # Create embedding for the query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    # Search top 3 matches
    distances, indices = index.search(query_embedding, k=3)
    
    # print("printing distances -----", distances)
    # print("printing indices -----", indices)
    
    # ================================================================================
    # printing distances ----- [[0.39902726 1.4865546  1.5680106 ]]
    # printing indices ----- [[0 3 1]]

    # ================================================================================
    # printing distances ----- [[0.8724602 1.762321  1.9198928]]
    # printing indices ----- [[4 1 3]]

    # ================================================================================
    # printing distances ----- [[0.80263966 1.6812193  1.8070664 ]]
    # printing indices ----- [[3 0 2]]

    # print("Top matches:")
    for rank, idx in enumerate(indices[0], start=1):
        print(f"{rank}. {documents[idx]}  (distance: {distances[0][rank-1]:.4f})")
        
    # ================================================================================
    # 1. FastAPI generates API documentation  (distance: 0.3990)
    # 2. LangChain helps build LLM applications  (distance: 1.4866)
    # 3. Python supports object oriented programming  (distance: 1.5680)

    # ================================================================================
    # 1. FAISS performs vector similarity search  (distance: 0.8725)
    # 2. Python supports object oriented programming  (distance: 1.7623)
    # 3. LangChain helps build LLM applications  (distance: 1.9199)

    # ================================================================================
    # 1. LangChain helps build LLM applications  (distance: 0.8026)
    # 2. FastAPI generates API documentation  (distance: 1.6812)
    # 3. Football is a popular sport  (distance: 1.8071)