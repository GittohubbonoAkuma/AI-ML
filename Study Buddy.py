from pdf2image import convert_from_path
import pytesseract
import numpy as np
import cv2
import streamlit as st
import tempfile
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import base64

st.set_page_config(page_title="Study Buddy",layout="centered",page_icon='📘')






@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model=load_model()


#
def pdf_to_img(path,dpi=300):
    images=convert_from_path(path,dpi=dpi)
    return images


#process image
def preprocess_image(pil_image):
    img=np.array(pil_image)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray=cv2.resize(
        gray,None,
        fx=2.0,fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )
    

    gray=cv2.fastNlMeansDenoising(gray,h=10)


    return gray


#convert PIL image to OpenCV format 
def ocr_image(preprocessed_img):
    custom_config=r'--oem 3 --psm 6'
    text=pytesseract.image_to_string(
        preprocessed_img,
        lang='eng',
        config=custom_config

    )
    return text


def extract_text_from_handwritten_pdf(path):
    images=pdf_to_img(path,dpi=300)
    full_text=""

    for i,img in enumerate(images):
        

        processed=preprocess_image(img)

        #cv2.imwrite(f"debug_page_{i+1}.png",processed)

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




#model=SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(chunks,query):
    embeddings=model.encode(chunks)
    query_embedding=model.encode([query])

    scores=cosine_similarity(query_embedding,embeddings)
    best_idx=scores[0].argmax()

    return chunks[best_idx]





def add_bg(image_file):
    with open(image_file,"rb") as f:
        encoded=base64.b64encode(f.read()).decode()
                                 
    st.markdown(
        f"""
        <style>
        .stApp{{
        background-image:url(data:image/jpg;base64,{encoded});
        background-size:cover;
        background-position:center;
        background-repeat:no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True

    )    
        



add_bg(r"lofi girl.jpg")


#streamlit UI
st.title("Study Buddy")
st.write("Upload your notes and ask your question")

uploaded_file=st.file_uploader("Upload PDF",type=["pdf"])

if uploaded_file is not None:
    if st.button("Process PDF"):
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())

            pdf_path=tmp.name

        with st.spinner("Running OCR....This may take some time"):
            raw_text=extract_text_from_handwritten_pdf(pdf_path)
            cleaned=clean_text(raw_text)
            chunks=chunk_text(cleaned)


        st.session_state["chunks"]=chunks

        st.success("PDF processed successfully!!")



if "chunks" in st.session_state:
            query=st.text_input("Ask a Question:")

            if query:
                answer=semantic_search(st.session_state["chunks"],query)
                st.subheader("Answer")
                st.write(answer)






#preprocessing might not work on all docs
