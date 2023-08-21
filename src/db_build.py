# File: db_build.py
import torch

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from constants import EMBEDDINGS_MODEL


# Device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load PDF file from data path
loader = DirectoryLoader('data/',
                         glob="*.pdf",
                         loader_cls=PyPDFLoader)
documents = loader.load()

# Split text from PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL,
                                   model_kwargs={'device': device})

# Build and persist FAISS vector store
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local('vectorstore/db_faiss')
