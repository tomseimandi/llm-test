# File: db_build.py
import argparse
from typing import Literal
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from constants import EMBEDDINGS_MODEL, DEVICE


def create_faiss_database(documents, embeddings):
    """Build and persist FAISS vector store.

    Args:
        database_name: Text documents.
        embeddings: Embeddings.
    """
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')


def create_chroma_database(documents, embeddings):
    """Build and persist Chroma vector store.

    Args:
        database_name: Text documents.
        embeddings: Embeddings.
    """
    vectorstore = Chroma.from_documents(documents, embeddings)
    vectorstore.save_local('vectorstore/db_chroma')


def main(database_name: Literal["faiss", "chroma"]):
    """Create FAISS of Chroma vector store for embeddings

    Args:
        database_name (Literal["faiss", "chroma"]): Database type.
    """
    # Load PDF file from data path
    loader = DirectoryLoader('data/',
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    raw_documents = loader.load()

    # Split text from PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL,
                                       model_kwargs={'device': DEVICE})
    if database_name == "faiss":
        create_faiss_database(documents, embeddings)
    elif database_name == "chroma":
        create_chroma_database(documents, embeddings)
    else:
        raise ValueError("Database name must be 'faiss' or 'chroma'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('database_name', type=str)
    args = parser.parse_args()

    main(database_name=args.database_name)
