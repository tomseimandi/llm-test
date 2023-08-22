# To update FAISS vector store
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from constants import EMBEDDINGS_MODEL, DEVICE


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL,
                                   model_kwargs={'device': DEVICE})


def update_vectorstore(docs, embeddings):
    faiss_db = FAISS.from_documents(docs, embeddings)

    if os.path.exists('vectorstore/db_faiss'):
        local_index = FAISS.load_local('vectorstore/db_faiss', embeddings)
        local_index.merge_from(faiss_db)
        local_index.save_local('vectorstore/db_faiss')
    else:
        faiss_db.save_local(folder_path='vectorstore/db_faiss')
