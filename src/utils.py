import streamlit as st
from langchain import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader


@st.cache_data
def parse_pdf(file):
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        output.append(text)

    return "\n\n".join(output)


@st.cache_data
def embed_text(text):
    """Split the text and embed it in a FAISS vector store"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=0, separators=["\n\n", ".", "?", "!", " ", ""]
    )
    texts = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    index = FAISS.from_texts(texts, embeddings)

    return index


@st.cache_data
def get_summary(text):
    model = OpenAI(temperature=0)
    summary_chain = load_summarize_chain(llm=model, chain_type="map_reduce")
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
    summary = summarize_document_chain.run(text)

    return summary


def get_answer(index, query):
    """Returns answer to a query using langchain QA chain"""

    docs = index.similarity_search(query)

    chain = load_qa_chain(OpenAI(temperature=0))
    answer = chain.run(input_documents=docs, question=query)

    return answer
