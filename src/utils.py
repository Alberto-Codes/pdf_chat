from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
import streamlit as st

@st.cache
def parse_pdf(file):
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        output.append(text)

    return "\n\n".join(output)

@st.cache
def embed_text(text):
    """Split the text and embed it in a FAISS vector store"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=0, separators=["\n\n", ".", "?", "!", " ", ""]
    )
    texts = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    index = FAISS.from_texts(texts, embeddings)

    return index