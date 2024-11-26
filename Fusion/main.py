#!/usr/bin/env python
"""
Author: Kavali Kranthi Kumar
Source: https://medium.com/thedeephub/rag-v-fusion-retrieval-82301e821424
requirements:
- langchain-openai
- numpy

"""


from langchain_openai.chat_models import ChatOpenAI

import numpy as np


from helper_functions import encode_pdf, encode_pdf_with_bm25


class FusionRetriever:

    def __init__(self, file_path, chunk_size=500, chunk_overlap=100):
        self.llm = ChatOpenAI(temperature=0, max_tokens=4000, model_name="gpt-4o-mini")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faiss_vectorstore = encode_pdf(file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.bm25_vectorstore = encode_pdf_with_bm25(file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    @staticmethod
    def fusion_algorithm(faiss_vectorstore, bm25_vectorstore, query, k=5, alpha=0.5):
        """
        Perform Fusion Retrival combining the vector_search and bm25_search results using a weighted sum.

        :param faiss_vectorstore: vectorstore useful for vector search
        :param bm25_vectorstore: vectorstore useful for bm25 search
        :param query: query to search
        :param k: number of documents to retrieve
        :param alpha: weight for vector search
        :return: list of top k documents
        """
        all_docs = faiss_vectorstore.similarity_search(query, k=faiss_vectorstore.index.ntotal)
        similar_docs = faiss_vectorstore.similarity_search_with_score(query, k=len(all_docs))
        bm25_scores = bm25_vectorstore.get_scores(query.split())

        vector_scores = np.array([score for doc, score in similar_docs])
        vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

        combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
        sorted_indices = np.argsort(combined_scores)[::-1]

        return [all_docs[i] for i in sorted_indices[:k]]

    def retrieve(self, query, k=3):
        top_docs = self.fusion_algorithm(self.faiss_vectorstore, self.bm25_vectorstore, query, k=k)
        return top_docs

