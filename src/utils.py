# File: utils.py
import torch
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from llm import llm
from prompts import qa_template
from constants import EMBEDDINGS_MODEL


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL,
                                   model_kwargs={'device': device})


# Wrap prompt template in a PromptTemplate object
def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt


# Build RetrievalQA object
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt},
                                       verbose=True)
    return dbqa


# Instantiate QA object
def setup_dbqa(): 
    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa


# Instantiate retriever
def retrieve(user_prompt):
    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    retriever = vectordb.as_retriever()
    return retriever.get_relevant_documents(user_prompt)


# Retrieve with filter
def retrieve_with_filter(user_prompt, source):
    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    retriever = vectordb.as_retriever(
        search_kwargs={"filter": {"source": source}}
    )
    return retriever.get_relevant_documents(user_prompt)


# Build RetrievalQA object with filter
def build_retrieval_qa_filter(llm, prompt, vectordb, source):
    retriever = vectordb.as_retriever(
        search_kwargs={"filter": {"source": source}, "k": 2}
    )
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt},
                                       verbose=True)
    return dbqa


# Filter dbqa
def setup_dbqa_filter(source):
    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa_filter(llm, qa_prompt, vectordb, source)

    return dbqa
