#!/usr/bin/env python

"""
Author: Kavali Kranthi Kumar

requirements:
- streamlit
"""

import streamlit as st

from Hyde.main import HyDeRetriever
from Basic.main import BasicRetriever


st.title("RAG retrieval techniques")

selected_technique = st.sidebar.selectbox("Select the retrieval technique", ["HyDe", "Basic"])

uploaded_file = st.file_uploader("Upload a PDF file.", type=["pdf"])
if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    if selected_technique == "HyDe":
        retriever = HyDeRetriever(uploaded_file.name)
        query = st.text_input("Enter the query:")
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
        query = st.text_input("Enter the query:")
        if query:
            similar_docs = retriever.retrieve(query)
            st.markdown("## :green[Retrieved documents:]")
            for i, doc in enumerate(similar_docs):
                st.write(f"Document {i+1}:")
                st.write(doc.page_content)
                st.write("----")
