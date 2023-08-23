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
DATABASE_NAME = 'chroma'

GPU_LAYERS = 35

# Paths
CHROMA_LOCAL_PATH = 'vectorstore/db_chroma'
S3_BUCKET_NAME = 'tseimandi'
S3_CHROMA_PREFIX = 'document_qa/vectorstore/db_chroma/'
S3_PDF_PREFIX = 'document_qa/pdf/'
