import streamlit as st

from utils import embed_text, get_answer, get_summary, parse_pdf

st.header("Doc QA")
uploaded_file = st.file_uploader("Upload a pdf", type=["pdf"])

if uploaded_file is not None:
    st.write("Summary: ")
    summary = get_summary(parse_pdf(uploaded_file))
    st.write(summary)
    index = embed_text(parse_pdf(uploaded_file))
    query = st.text_area("Ask a question about the document")
    button = st.button("Submit")
    if button:
        st.write(get_answer(index, query))
