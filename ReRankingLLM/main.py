#!/usr/bin/env python

"""
Author: Kavali Kranthi Kumar
Source: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/reranking.ipynb
Requirements:
- langchain-openai
- langchain
- pydantic
- python-dotenv

helper functions are used to parse, embed and process the pdf file
"""
import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field

from helper_functions import encode_pdf

from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# set openai api key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")


class ReRankingLLM:

    def __init__(self, file_path, chunk_size=500, chunk_overlap=100):
        self.llm = ChatOpenAI(temperature=0, max_tokens=4000, model_name="gpt-4o-mini")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.rerank_prompt = ChatPromptTemplate.from_messages([("system", """On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:""")])
        self.rerank_chain = self.rerank_prompt | self.llm.with_structured_output(RatingScore)

    def rerank(self, query, documents, k=5):
        """
        Perform Re-Ranking using the vector_search results.
        :param query: The User Query
        :param documents: The list of documents to re-rank, which are initially retrieved using vector search
        :param k: The number of documents to return
        :return: The top k documents
        """
        scored_docs = []
        for doc in documents:
            input_variables = {"query": query, "doc": doc.page_content}
            score = self.rerank_chain.invoke(input_variables).relevance_score
            try:
                score = float(score)
            except ValueError:
                score = 0  # Default score if parsing fails
            print(f"score: {score}\n doc: {doc.page_content}")
            scored_docs.append((doc, score))
        re_ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in re_ranked_docs[:k]]

    def retrieve(self, query, k=5):
        documents = self.vectorstore.similarity_search(query, k=k)
        top_docs = self.rerank(query, documents, k=k)
        return documents, top_docs
