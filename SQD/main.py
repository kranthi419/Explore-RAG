#!/usr/bin/env python

"""
Author: Kavali Kranthi Kumar
Source: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/query_transformations.ipynb
Requirements:
- langchain-openai
- langchain-core
- python-dotenv

helper functions are used to parse, embed and process the pdf file
"""

import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


from helper_functions import encode_pdf

from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# set openai api key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class SubQueryDecompositionRetriever:

    def __init__(self, file_path, chunk_size=500, chunk_overlap=100):
        self.llm = ChatOpenAI(temperature=0, max_tokens=4000, model_name="gpt-4o-mini")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.sub_query_decomposition_prompt = ChatPromptTemplate.from_messages([("system", """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
        Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.
        
        Original query: {original_query}
        
        example: What are the impacts of climate change on the environment?
        
        Sub-queries:
        1. What are the impacts of climate change on biodiversity?
        2. How does climate change affect the oceans?
        3. What are the effects of climate change on agriculture?
        4. What are the impacts of climate change on human health?""")])
        self.sub_query_decomposition_chain = self.sub_query_decomposition_prompt | self.llm

    def decompose_question_to_sub_queries(self, query):
        input_variables = {"original_query": query}
        response = self.sub_query_decomposition_chain.invoke(input_variables).content
        return [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]

    def retrieve(self, query, k=3):
        sub_questions = self.decompose_question_to_sub_queries(query=query)
        for question in sub_questions:
            similar_docs = self.vectorstore.similarity_search(question, k=k)
            yield question, similar_docs


