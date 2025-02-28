#!/usr/bin/env python

"""
Author: Kavali Kranthi Kumar
Source: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/reliable_rag.ipynb
Requirements:
- langchain-openai
- langchain-core
- python-dotenv
"""

from dotenv import load_dotenv, find_dotenv

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# load environment variables from .env file
load_dotenv(find_dotenv())

embedding_model = OpenAIEmbeddings()

# Docs to index
urls = [
    "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
]

# load the documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# chunking
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=0
)

docs_splits = text_splitter.split_documents(docs_list)

# adding to vector store
vectorstore = Chroma.from_documents(documents=docs_splits, collection_name="rag", embedding=embedding_model)
retriever = vectorstore.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 4})

question = input("enter the question: ")
docs = retriever.invoke(question)
# print("retrieved documents: ", docs)
# print(f"Title: {docs[0].metadata['title']}\n\nSource: {docs[0].metadata['source']}\n\nContent: {docs[0].page_content}\n")


# checking document relevancy
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI


class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

docs_to_use = []
for doc in docs:
    print(doc.page_content, '\n', '-'*50)
    res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    print(res, '\n')
    if res.binary_score == 'yes':
        docs_to_use.append(doc)



















