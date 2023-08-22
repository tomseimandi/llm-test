# File: llm.py
# from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from constants import GPU_LAYERS


# Local CTransformers wrapper for Llama-2-7B-Chat
# llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
#                     model_type='llama',  # Model type Llama
#                     config={'max_new_tokens': 256,
#                             'temperature': 0.01},
#                     gpu_layers=GPU_LAYERS)


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager
n_batch = 512
llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
    n_gpu_layers=GPU_LAYERS,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,
)
