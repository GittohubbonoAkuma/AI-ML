from pypdf import PdfReader
from sentence_transformer import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model=SentenceTransformer('all-MiniLM-L6-v2')
#function to read the pdf
def pdf_reader(path):
  reader=PdfReader(path)
  text=""
  for page in reader.pages:
    text+=page.extract_text()
  return text
text=pdf_reader(path)

#create chunks
def chunk_text(text,chunk_size=100):
  words=text.split()
  chunks=[]
  for i in range(0,len(words),chunk_size):
    chunk="".join(words[i:i+chunk_size])
    chunks.append(chunk)
  return chunks
chunks=chunk_text(text)


question=input("What do you want to learn?")
#create embeddings
chunk_embeddings=model.encode([chunks])

question_embeddings=model.encode([question])


#check similarities
similarities=cosine_similarity(question_embeddings,chunk_embeddings)

best=similarities[0].argmax()
print("Here's your answer")
print(chunks[best])
