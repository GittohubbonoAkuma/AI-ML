from sentence_transformer import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model=SentenceTransformer('all-MiniLM-l6-v2')

#input sentences
sentence1=input("Enter your first sentence")
sentence2=input("Enter your second sentence")

#create embeddings
embedding=model.encode([sentence1,sentence2])
similarity=cosine_similarity([embedding[0]],[embedding[1]])
print(f"Similarity fraction={similarity}")
