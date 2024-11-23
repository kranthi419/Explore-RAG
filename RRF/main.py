#!/usr/bin/env python
"""
Author: Kavali Kranthi Kumar
Source: https://gist.github.com/srcecde/eec6c5dda268f9a58473e1c14735c7bb
Learn more about Reciprocal Rank Fusion:
https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a
https://safjan.com/implementing-rank-fusion-in-python/
requirements:
- langchain-openai

"""

from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import Document

from helper_functions import encode_pdf
from collections import defaultdict


class RRFRetriever:

    def __init__(self, file_path, chunk_size=500, chunk_overlap=100):
        self.llm = ChatOpenAI(temperature=0, max_tokens=4000, model_name="gpt-4o-mini")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    @staticmethod
    def reciprocal_rank_fusion(*list_of_list_ranks_system, k=60):
        """
        Fuse rank from multiple IR systems using Reciprocal Rank Fusion.
        Args:
        * list_of_list_ranks_system: Ranked results from different IR system.
        k (int): A constant used in the RRF formula (default is 60).

        Returns:
        Tuple of list of sorted documents by score and sorted documents
        """
        # Dictionary to store RRF mapping
        rrf_map = defaultdict(float)
        # Calculate RRF score for each result in each list
        for rank_list in list_of_list_ranks_system:
            for rank, item in enumerate(rank_list, 1):
                rrf_map[item] += 1 / (rank + k)
        # Sort items based on their RRF scores in descending order
        sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
        # Return tuple of list of sorted documents by score and sorted documents
        return sorted_items, [item for item, score in sorted_items]

    def process_docs(self, docs):
        processed_docs = []
        for doc in docs:
            processed_docs.append(doc.page_content)
        return processed_docs

    def retrieve(self, query, k=3):
        similar_docs = self.process_docs(self.vectorstore.similarity_search(query, k=k))
        mmr_docs = self.process_docs(self.vectorstore.max_marginal_relevance_search(query, k=k))
        # Combine the results of documents from different algorithms using Reciprocal Rank Fusion
        re_ranked_docs_with_scores, re_ranked_docs = self.reciprocal_rank_fusion(similar_docs, mmr_docs)
        final_docs = [Document(page_content=doc) for doc in re_ranked_docs]
        return final_docs
