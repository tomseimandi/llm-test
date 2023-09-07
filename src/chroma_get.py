# File: chroma_get.py
from chroma_s3 import ChromaS3
from constants import (
    S3_BUCKET_NAME,
    S3_CHROMA_PREFIX,
    CHROMA_LOCAL_PATH
)
from db_utils import embeddings
from pathlib import Path


if __name__ == "__main__":
    vectordb = ChromaS3.from_s3(
        S3_BUCKET_NAME,
        S3_CHROMA_PREFIX,
        CHROMA_LOCAL_PATH,
        embedding_function=embeddings
    )
    sources = vectordb.get(include=["metadatas"]).get("metadatas")
    sources = [tuple(Path(source.get("source")).stem.split("_")) for source in sources]
    available_documents = list(set(sources))
    print(available_documents)
    