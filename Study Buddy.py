from pdf2image import convert_from_path
import pytesseract
import numpy as np
import cv2
import tempfile
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import base64

st.set_page_config(page_title="Study Buddy",layout="wide",icon='📘')

@st.cache_resource
def load_model():
  model=SentenceTransformer("all-MiniLM-L6-v2")

model=load_model()

def pdf_to_img(pat,dpi=300):
  images=convert_from_path(path,dpi=dpi)
  return images


pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#process image

def preprocess_img(pil_image):
  img=np.array(pil_image)

  gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  gray_img=cv2.resize(
  gray_img,None,fx=2.0,fy=2.0,interpolation=cv2.INTER_CUBIC)


  gray_img=cv2.fastNlMeansDenoising(gray,h=10)

  return gray_img


#Convert PIL image to OpenCV format
def ocr_image(preprocessed_img):
  custom_config=r'--oem 3 --psm 6'
  text=pytesseract.image_to_string(
  preprocessed_img,
  lang="eng",
  config=custom_config)

  return text
def extract_text(path):
  images=pdf_to_img(path,dpi=300)
  full_text=""
  for i,img in enumerate(images):

    processed=preprocess_image(img)
    page_text=ocr_image(processed)
    full_text+=page_text+"\n"

  return full_text
def clean_text(text):
  text=text.replace("\n"," ")
  text=" ".join(text.split())

  return text



def chunk_text(text,chunk_size=50):
  words=text.split()
  return[
    " ".join(words[i:i+chunk_size])
    for i in range(0,len(words),chunk_size)
  ]


def semantic_search(chunks,query):
  embeddings=model.encode(chunks)
  query_embeddings=model.encode([query])

  scores=cosine_similarity(query_embeddings,embeddings)
  best=scores[0].argmax()

  return chunks[best]

def add_bg(image_path):
  with open(image_file,"rb) as f:
    encoded=base64.b64encode(f.read()).decode()

  st.markdown(
    f"""
    <style
    .stApp{{
    background-image:url(data:image/jpg;base64,{encoded});
    background-size:cover;
    background-repeat:no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True

  )

#Added my personal bg(lo-fi girl)
add_bg(r"C:\Users\susan\Downloads\lofi girl.jpg")

st.title("Study Buddy")
st.write("Upload your notes and ask your question")

uploaded_fies=st.file_uploader("Upload PDF",type=["pdf"])

if uploaded_file is not None:
  if st.button("Process PDF"):
    with tempfile.NamedTemporaryFile(delete=false,suffix=".pdf") as tmp:
      tmp.write(uploaded_file.read())

      pdf_pth=tmp.name

    with st.spinner("Running OCR.....Please wait")
      raw_text=extract_text_from_pdf(pdf_path)
      cleaned=clean_text(raw_text)
      chunks=chunk_text(cleaned)


  st.session_state["chunks"]=chunks
  st.success("PDF processed successfully!!")


if "chunks" in st.session_state:
  query=st.text_input("What do you want to learn?")

  if query:
    answer=semantic_search(st.session_state["chunks"],query)
    st.subheader("Answer")
    st.write(answer)
    
