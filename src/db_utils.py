# To update FAISS vector store
import os
import tempfile
import boto3
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from constants import EMBEDDINGS_MODEL, DEVICE, CHROMA_LOCAL_PATH
from chroma_s3 import ChromaS3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredPDFLoader


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL,
                                   model_kwargs={'device': DEVICE})


def create_faiss_vectorstore(documents, embeddings):
    """Build and persist FAISS vector store.

    Args:
        documents: Text documents.
        embeddings: Embeddings.
    """
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')


def create_chroma_from_documents(documents, embeddings):
    """Build and persist Chroma vector store from documents.

    Args:
        documents: Text documents.
        embeddings: Embeddings.
    """
    vectorstore = ChromaS3.from_documents(
        documents,
        embeddings,
        persist_directory=CHROMA_LOCAL_PATH
    )
    return vectorstore


def create_chroma(embeddings):
    """Build and persist Chroma vector store.

    Args:
        embeddings: Embeddings.
    """
    vectorstore = ChromaS3(
        embedding_function=embeddings,
        persist_directory=CHROMA_LOCAL_PATH
    )
    return vectorstore


def add_docs_to_faiss_vectorstore(docs, embeddings):
    faiss_db = FAISS.from_documents(docs, embeddings)

    if os.path.exists('vectorstore/db_faiss'):
        local_index = FAISS.load_local('vectorstore/db_faiss', embeddings)
        local_index.merge_from(faiss_db)
        local_index.save_local('vectorstore/db_faiss')
    else:
        faiss_db.save_local(folder_path='vectorstore/db_faiss')
    return


def update_chroma(
    chroma_vectorstore: ChromaS3,
    s3_bucket_name: str,
    s3_prefix: str
):
    """
    Get all pdf files with the specified s3 prefix,
    update the vectorstore to include embedding of
    those files.

    TODO: for now the PDF files are supposed to be
    readable, add OCR case.

    Args:
        chroma_vectorstore (ChromaS3): Vectorstore.
        s3_bucket_name (str): The name of the S3 bucket.
        s3_prefix (str): Desired prefix in S3 bucket.
    """
    s3 = boto3.client(
        "s3",
        endpoint_url='https://'+'minio.lab.sspcloud.fr'
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        response = s3.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_prefix)
        for obj in response.get('Contents', []):
            s3_key = obj['Key']
            if s3_key.lower().endswith('.pdf'):
                local_file_path = os.path.join(
                    temp_dir,
                    os.path.basename(s3_key)
                )
                s3.download_file(s3_bucket_name, s3_key, local_file_path)

        loader = DirectoryLoader(
            temp_dir,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        raw_documents = loader.load()

        # Split text from PDF into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        documents = text_splitter.split_documents(raw_documents)
        for document in documents:
            clean_path = os.path.basename(document.metadata['source'])
            document.metadata['source'] = clean_path
        chroma_vectorstore.add_documents(documents)
    return


def update_chroma_unstructured(
    chroma_vectorstore: ChromaS3,
    s3_bucket_name: str,
    s3_prefix: str
):
    # Files that are already in the vector store
    metadatas = chroma_vectorstore.get().get('metadatas')
    files = [metadata['source'] for metadata in metadatas]
    unique_files = list(set(files))
    print(f'{len(unique_files)} already in the vector store.')

    s3 = boto3.client(
        "s3",
        endpoint_url='https://'+'minio.lab.sspcloud.fr'
    )

    raw_documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        response = s3.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_prefix)
        n_added_files = 0
        for obj in response.get('Contents', []):
            s3_key = obj['Key']
            if (s3_key.lower().endswith('.pdf')) and (os.path.basename(s3_key) not in unique_files):
                n_added_files += 1
                local_file_path = os.path.join(
                    temp_dir,
                    os.path.basename(s3_key)
                )
                s3.download_file(s3_bucket_name, s3_key, local_file_path)

                loader = UnstructuredPDFLoader(
                    local_file_path
                )
                raw_documents += loader.load()
        print(f"{n_added_files} new file(s) added to the vector store.")

        # Split text from PDF into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        documents = text_splitter.split_documents(raw_documents)
        for document in documents:
            clean_path = os.path.basename(document.metadata['source'])
            document.metadata['source'] = clean_path
        chroma_vectorstore.add_documents(documents)
    return
