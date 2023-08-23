from __future__ import annotations

from typing import Type
import boto3
import os
import shutil
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings


class ChromaS3(Chroma):

    def save_to_s3(self, s3_bucket_name, s3_prefix):
        """
        Save the vector store to an S3 bucket.

        Args:
            s3_bucket_name (str): The name of the S3 bucket.
            s3_prefix (str): Desired prefix in S3 bucket.
        """
        # Create a connection to S3
        s3 = boto3.client(
            "s3",
            endpoint_url='https://'+'minio.lab.sspcloud.fr'
        )

        for root, dirs, files in os.walk(self._persist_directory):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_key = s3_prefix + os.path.relpath(
                    local_file_path,
                    self._persist_directory
                )
                s3.upload_file(local_file_path, s3_bucket_name, s3_key)

    @classmethod
    def from_s3(
        cls: Type[ChromaS3],
        s3_bucket_name: str,
        s3_prefix: str,
        persist_directory: str,
        embedding_function: Embeddings
    ) -> ChromaS3:
        """
        Create persistent Chroma vectorstore from s3 files.

        Args:
            s3_bucket_name (str): The name of the S3 bucket.
            s3_prefix (str): Desired prefix in S3 bucket.
            persist_directory (str): Local persistence.
            embedding_function (Embeddings): Embedding function.

        Returns:
            ChromaS3: ChromaS3 object.
        """
        # If persist_directory exists, delete it and its content
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"Folder '{persist_directory}' and its content "
                  f"successfully deleted.")

        s3 = boto3.client(
            "s3",
            endpoint_url='https://'+'minio.lab.sspcloud.fr'
        )

        response = s3.list_objects_v2(
            Bucket=s3_bucket_name,
            Prefix=s3_prefix
        )
        for obj in response.get('Contents', []):
            s3_key = obj['Key']
            local_file_path = os.path.join(
                persist_directory,
                os.path.relpath(s3_key, s3_prefix)
            )
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3.download_file(s3_bucket_name, s3_key, local_file_path)
            print(
                f"Downloaded s3://{s3_bucket_name}/{s3_key} to {local_file_path}"
            )

        return ChromaS3(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
