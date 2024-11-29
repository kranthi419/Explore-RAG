#!/usr/bin/env python

"""
Author: Kavali Kranthi Kumar

requirements:
- streamlit
"""
import os

import streamlit as st

from Hyde.main import HyDeRetriever
from Basic.main import BasicRetriever
from RRF.main import RRFRetriever
from Fusion.main import FusionRetriever
from SQD.main import SubQueryDecompositionRetriever
from ReRankingLLM.main import ReRankingLLM


st.set_page_config(layout="wide")

st.title("RAG retrieval techniques")

retrival_techniques_files_path = {"HyDe": "Hyde/main.py", "Basic": "Basic/main.py", "RRF": "RRF/main.py", "Fusion": "Fusion/main.py",
                                  "SQD": "SQD/main.py", "ReRankingLLM": "ReRankingLLM/main.py"}
selected_technique = st.sidebar.selectbox("Select the retrieval technique", ["HyDe", "Basic", "RRF", "Fusion", "SQD", "ReRankingLLM"])
uploaded_file = st.sidebar.file_uploader("Upload a PDF file.", type=["pdf"])
st.sidebar.caption("Upload the PDF file and ask a query to see the results.")

if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    if selected_technique == "HyDe":
        retriever = HyDeRetriever(uploaded_file.name)
        query = st.text_input("**Enter the query:**")
        if query:
            similar_docs, hypothetical_doc = retriever.retrieve(query)
            st.markdown("## :green[Hypothetical document:]")
            st.write(hypothetical_doc)
            st.markdown("## :green[Retrieved documents:]")
            for i, doc in enumerate(similar_docs):
                st.write(f"Document {i+1}:")
                st.write(doc.page_content)
                st.write("----")
    elif selected_technique == "Basic":
        retriever = BasicRetriever(uploaded_file.name)
        query = st.text_input("**Enter the query:**")
        if query:
            similar_docs = retriever.retrieve(query)
            st.markdown("## :green[Retrieved documents:]")
            for i, doc in enumerate(similar_docs):
                st.write(f"Document {i+1}:")
                st.write(doc.page_content)
                st.write("----")
    elif selected_technique == "RRF":
        retriever = RRFRetriever(uploaded_file.name)
        query = st.text_input("**Enter the query:**")
        if query:
            similar_docs = retriever.retrieve(query)
            st.markdown("## :green[Retrieved documents:]")
            for i, doc in enumerate(similar_docs):
                st.write(f"Document {i+1}:")
                st.write(doc.page_content)
                st.write("----")
    elif selected_technique == "Fusion":
        retriever = FusionRetriever(uploaded_file.name)
        query = st.text_input("**Enter the query:**")
        if query:
            similar_docs = retriever.retrieve(query)
            st.markdown("## :green[Retrieved documents:]")
            for i, doc in enumerate(similar_docs):
                st.write(f"Document {i+1}:")
                st.write(doc.page_content)
                st.write("----")
    elif selected_technique == "SQD":
        retriever = SubQueryDecompositionRetriever(uploaded_file.name)
        query = st.text_input("**Enter the query:**")
        if query:
            for question, similar_docs in retriever.retrieve(query):
                st.markdown(f"## :green[Question:] {question}")
                st.markdown("## :green[Retrieved documents:]")
                for i, doc in enumerate(similar_docs):
                    st.write(f"Document {i+1}:")
                    st.write(doc.page_content)
                    st.write("----")
    elif selected_technique == "ReRankingLLM":
        retriever = ReRankingLLM(uploaded_file.name)
        query = st.text_input("**Enter the query:**")
        if query:
            similar_docs, rerank_docs = retriever.retrieve(query)
            st.markdown("## :green[Retrieved documents:]")
            for i, doc in enumerate(similar_docs):
                st.write(f"Document {i+1}:")
                st.write(doc.page_content)
                st.write("----")
            st.markdown("## :green[Re-ranked documents:]")
            for i, doc in enumerate(rerank_docs):
                st.write(f"Document {i+1}:")
                st.write(doc.page_content)
                st.write("----")
    os.remove(uploaded_file.name)
else:
    file_path = retrival_techniques_files_path[selected_technique]
    with open(file_path) as f:
        code = f.read()
    st.markdown("## :green[Code snippet:]")
    st.code(code, language="python")
