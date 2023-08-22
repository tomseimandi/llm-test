# Constants
import torch


EMBEDDINGS_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# EMBEDDINGS_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Device
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# Database name for the vector store
DATABASE_NAME = 'faiss'

GPU_LAYERS = 35
