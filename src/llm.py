# File: llm.py
# from langchain.llms import CTransformers
from typing import Optional
import boto3
import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from constants import GPU_LAYERS, LLM_PATH, S3_BUCKET_NAME, S3_LLM_KEY
import streamlit as st
from stream_handler import StreamDisplayHandler


@st.cache_resource
def get_llm(_display_handler: Optional[StreamDisplayHandler] = None, from_s3: bool = True):
    if from_s3:
        download_llm_from_s3()

    # Local CTransformers wrapper for Llama-2-7B-Chat
    # llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
    #                     model_type='llama',  # Model type Llama
    #                     config={'max_new_tokens': 256,
    #                             'temperature': 0.01},
    #                     gpu_layers=GPU_LAYERS)

    # Callbacks support token-wise streaming
    if _display_handler is not None:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler(), _display_handler])
    else:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Verbose is required to pass to the callback manager
    n_batch = 512
    llm = LlamaCpp(
        model_path=LLM_PATH,
        n_gpu_layers=GPU_LAYERS,
        n_batch=n_batch,
        f16_kv=True,
        use_mlock=True,
        n_ctx=1024,
        n_threads=8,
        callback_manager=callback_manager,
        verbose=True,
        streaming=True
    )
    # HACK to test: https://twitter.com/RLanceMartin/status/1681879318493003776?s=20
    # https://python.langchain.com/docs/guides/local_llms

    # Llama.cpp was born to run models on CPUs, recently introduced
    # acceleration with the GPU, but it's just speeding up some
    # types of computation, it's not running the whole model on the
    # GPU, so llama.cpp it's still CPU intensive. So your problem
    # will definitely be limited by your CPU.
    return llm


def download_llm_from_s3():
    if not os.path.exists(LLM_PATH):
        s3 = boto3.client(
            "s3",
            endpoint_url='https://'+'minio.lab.sspcloud.fr'
        )
        s3.download_file(S3_BUCKET_NAME, S3_LLM_KEY, LLM_PATH)
    return
