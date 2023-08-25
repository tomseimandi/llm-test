# File: db_build.py
import argparse
from typing import Literal
from langchain.embeddings import HuggingFaceEmbeddings
from constants import (
    EMBEDDINGS_MODEL,
    DEVICE,
    S3_BUCKET_NAME,
    S3_CHROMA_PREFIX,
    S3_PDF_PREFIX
)
from db_utils import create_chroma, update_chroma_unstructured


def main(database_name: Literal["faiss", "chroma"]):
    """Create FAISS of Chroma vector store for embeddings

    Args:
        database_name (Literal["faiss", "chroma"]): Database type.
    """
    if database_name == "faiss":
        raise NotImplementedError("No implementation yet for FAISS.")
    elif database_name == "chroma":
        # Create locally persistent vectorstore from PDF stored on s3
        # and export it to s3 to make it available for later use
        vectorstore = create_chroma(
            embeddings=HuggingFaceEmbeddings(
                model_name=EMBEDDINGS_MODEL,
                model_kwargs={'device': DEVICE}
            )
        )

        # TODO: make this a method of class ChromaS3 ?
        update_chroma_unstructured(
            vectorstore,
            S3_BUCKET_NAME,
            S3_PDF_PREFIX)
        vectorstore.save_to_s3(
            S3_BUCKET_NAME,
            S3_CHROMA_PREFIX
        )
    else:
        raise ValueError("Database name must be 'faiss' or 'chroma'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('database_name', type=str)
    args = parser.parse_args()

    main(database_name=args.database_name)
