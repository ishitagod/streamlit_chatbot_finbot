from typing import Any
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader

import pymupdf  # PyMuPDF
from PIL import Image
import os
import re
import tempfile
import chromadb
import faiss
from huggingface_hub import login
from langchain.docstore.document import Document  

class MyApp:
    def __init__(self, hugging_face_api: str = None) -> None:
        self.hugging_face_api: str = hugging_face_api
        self.chain = None
        self.chat_history: list = []
        self.N: int = 0
        self.count: int = 0

    def __call__(self, file_path: str) -> Any:
        if self.count == 0:
            self.chain = self.build_chain(file_path)
            self.count += 1
        return self.chain
    def process_file_in_chunks(self, file: str, chunk_size: int = 5):
        doc = pymupdf.open(file)
        chunks = []
        for i in range(0, len(doc), chunk_size):
            text = ""
            for j in range(i, min(i + chunk_size, len(doc))):
                text += doc[j].get_text()
            chunks.append(Document(page_content=text))
        return chunks

    def process_file(self, file: str):
        loader = PyPDFLoader(file)
        documents = loader.load()
        pattern = r"/([^/]+)$"
        match = re.search(pattern, file)
        try:
            file_name = match.group(1)
        except:
            file_name = os.path.basename(file)

        return documents, file_name

    def build_chain(self, file_path: str):
        chunks = self.process_file_in_chunks(file_path)
        documents = [{"text": chunk} for chunk in chunks]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
        pdfsearch = FAISS.from_documents(chunks, embeddings)
        
        chain = ConversationalRetrievalChain.from_llm(
            HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=self.hugging_face_api),
            retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True,
        )
        return chain

def get_response(app, query, file_path):
    if not file_path:
        st.error("Upload a PDF")
        return
    chain = app(file_path)
    result = chain(
        {"question": query, "chat_history": app.chat_history}, return_only_outputs=True
    )
    app.chat_history += [(query, result["answer"])]
    app.N = list(result["source_documents"][0])[1][1]
    return result["answer"]

def render_file(file_path, page_num):
    doc = pymupdf.open(file_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=150)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image

def purge_chat(app):
    app.chat_history = []
    app.count = 0

st.set_page_config(page_title="FinBot")
st.title("💬 FinBot")
st.caption("🚀 LLM + RAG + Langchain powered!")
page_bg_img = '''
    <style>
        .stApp{
            background-image: url("https://images.unsplash.com/photo-1483791424735-e9ad0209eea2?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
        }
        [data-testid="stBottom"] > div {
            background: transparent;
        }
    </style>
    '''
st.markdown(page_bg_img,unsafe_allow_html=True)

huggingfacehub_api = "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"
login(huggingfacehub_api)
app = MyApp(huggingfacehub_api)
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
            st.session_state['tmp_file_path'] = tmp_file_path

    if st.button("Submit"):
        if 'tmp_file_path' not in st.session_state:
            st.warning("Please upload the document.")
        else:
            purge_chat(app)
            tmp_file_path = st.session_state['tmp_file_path']
            st.image(render_file(tmp_file_path, 0), caption="Page 1")
            with st.spinner("Processing the document..."):
                try:
                    app(tmp_file_path)
                    st.session_state['chain_built'] = True
                    st.success("Document processed and embeddings generated successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

query = st.text_input("Enter your query")
if query and 'tmp_file_path' in st.session_state and 'chain_built' in st.session_state:
    with st.spinner("Generating response..."):
        response = get_response(app, query, st.session_state['tmp_file_path'])
        if response:
            pattern = r"Helpful Answer: (.+)"
            match = re.search(pattern, response)
            helpful_answer = match.group(1) if match else response
            
            st.write(f"Response: {helpful_answer}")
