import os
from dotenv import load_dotenv
from openai import AsyncOpenAI, AsyncAzureOpenAI

load_dotenv()

# --- Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
EMBEDDING_MODEL_URL = os.getenv("EMBEDDING_MODEL_URL", OPENAI_BASE_URL)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")
if not OPENAI_BASE_URL:
    raise ValueError("OPENAI_BASE_URL not found in .env file")

# --- Clients ---
# llm_client = AsyncAzureOpenAI(
#     api_version="2024-12-01-preview",
#     api_key=OPENAI_API_KEY,
#     azure_endpoint=OPENAI_BASE_URL,
# )
llm_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

embedding_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=EMBEDDING_MODEL_URL)

# --- Model Names ---
# LLM_MODEL_NAME = "gpt4o"
LLM_MODEL_NAME = "Qwen3-8B"
EMBEDDING_MODEL_NAME = "Qwen3-Embedding-0.6B"
# EMBEDDING_MODEL_NAME = "text-embedding-qwen3-embedding-0.6b"

# --- Graph and Text Configuration ---
GRAPH_OUTPUT_PATH = "output/knowledge_graph.gml"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 0

# --- Milvus Configuration ---
MILVUS_URI = "./milvus_data/milvus.db"
COLLECTION_NAME = "knowledge_graph_chunks"
VECTOR_DIMENSION = 1024  # For text-embedding-qwen3-embedding-0.6b
