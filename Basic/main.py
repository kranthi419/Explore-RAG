#!/usr/bin/env python

"""
Author: Kavali Kranthi Kumar
Source: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag.ipynb
requirements:
- langchain-openai
- python-dotenv

helper functions are used to parse, embed and process the pdf file
"""

import os
from langchain_openai.chat_models import ChatOpenAI

from helper_functions import encode_pdf, show_context

from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# set openai api key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class BasicRetriever:

    def __init__(self, file_path, chunk_size=500, chunk_overlap=100):
        self.llm = ChatOpenAI(temperature=0, max_tokens=4000, model_name="gpt-4o-mini")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def retrieve(self, query, k=3):
        similar_docs = self.vectorstore.similarity_search(query, k=k)
        return similar_docs


if __name__ == "__main__":
    filepath = "your_sample.pdf"
    retriever = BasicRetriever(filepath)
    question = "What is the capital of France?"
    similar_docs_data = retriever.retrieve(question)
    docs_content = [doc.content for doc in similar_docs_data]
    show_context(docs_content)
