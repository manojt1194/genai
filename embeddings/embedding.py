from sentence_transformers import (
    SentenceTransformer,
    util
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

documents = [
    "FastAPI generates API documentation",
    "Python supports object oriented programming",
    "Football is a popular sport"
]

doc_embeddings = model.encode(documents)

# print("printing doc embedding -----",doc_embeddings)


query = "How does FastAPI create docs?"

query_embedding = model.encode(query)

# print("printing query_embedding -----",query_embedding)

scores = util.cos_sim(
    query_embedding,
    doc_embeddings
)

print("Printing score ------------------------",scores)

print("Max score index -------", scores.argmax())