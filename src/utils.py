# File: utils.py
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from llm import llm
from prompts import qa_template
from constants import EMBEDDINGS_MODEL, DEVICE, DATABASE_NAME


# Wrap prompt template in a PromptTemplate object
def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt


# Build RetrievalQA object
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt},
        verbose=True
    )
    return dbqa


# Instantiate QA object
def setup_dbqa():
    vectordb = load_vectorstore(DATABASE_NAME)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa


# Instantiate retriever
def retrieve(user_prompt):
    vectordb = load_vectorstore(DATABASE_NAME)
    retriever = vectordb.as_retriever()
    return retriever.get_relevant_documents(user_prompt)


# Retrieve with filter
def retrieve_with_filter(user_prompt, source):
    vectordb = load_vectorstore(DATABASE_NAME)
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
    vectordb = load_vectorstore(DATABASE_NAME)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa_filter(llm, qa_prompt, vectordb, source)

    return dbqa


def load_vectorstore(database_name):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL,
                                       model_kwargs={'device': DEVICE})
    if database_name == "faiss":
        return load_faiss_vectorstore(database_name, embeddings)
    elif database_name == "chroma":
        return load_chroma_vectorstore(database_name, embeddings)
    else:
        raise ValueError("Database name must be 'faiss' or 'chroma'")


def load_faiss_vectorstore(database_name, embeddings):
    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    return vectordb


def load_chroma_vectorstore(database_name, embeddings):
    vectordb = Chroma(
        persist_directory='vectorstore/db_chroma',
        embedding_function=embeddings
    )
    return vectordb
