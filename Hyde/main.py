#!/usr/bin/env python

"""
Author: Kavali Kranthi Kumar
Source: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb
requirements:
- langchain-openai
- langchain-core
- python-dotenv

helper functions are used to parse, embed and process the pdf file
"""

import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from helper_functions import encode_pdf, text_wrap, show_context

from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# set openai api key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# define the HyDe retriever class - creating vector store, generating hypothetical document, and retrieving
class HyDeRetriever:
    def __init__(self, file_path, chunk_size=500, chunk_overlap=100):
        self.llm = ChatOpenAI(temperature=0, max_tokens=4000, model_name="gpt-4o-mini")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.hyde_prompt = ChatPromptTemplate.from_messages([("system", """Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
            The document size has to be exactly {chunk_size} characters.""")])
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
        return similar_docs, hypothetical_doc


if __name__ == "__main__":
    filepath = "your_sample.pdf"

    retriever = HyDeRetriever(filepath)
    question = "What is the capital of France?"
    similar_docs_data, hypothetical_docs_data = retriever.retrieve(question)

    docs_content = [doc.content for doc in similar_docs_data]
    print("hypothetical_doc:\n")
    print(text_wrap(hypothetical_docs_data)+"\n")
    show_context(docs_content)
