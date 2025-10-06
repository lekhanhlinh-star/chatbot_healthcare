from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Try to load from local cache first, fallback to download if not available
import os
model_path = "./models/bge-reranker-v2-m3"
if os.path.exists(model_path):
    reranker = HuggingFaceCrossEncoder(model_name=model_path)