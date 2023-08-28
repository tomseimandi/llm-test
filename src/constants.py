# Constants
import torch


EMBEDDINGS_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# EMBEDDINGS_MODEL = 'antoinelouis/biencoder-camembert-base-mmarcoFR'
# EMBEDDINGS_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Device
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# Database name for the vector store
DATABASE_NAME = 'chroma'

GPU_LAYERS = 0

# Paths
CHROMA_LOCAL_PATH = 'vectorstore/db_chroma'
S3_BUCKET_NAME = 'tseimandi'
S3_CHROMA_PREFIX = 'document_qa/vectorstore/db_chroma/'
S3_PDF_PREFIX = 'document_qa/pdf/'
LLM_PATH = 'models/llama-2-7b-chat.ggmlv3.q4_K_M.bin'
