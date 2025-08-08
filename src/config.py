import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Model configuration
LLM_MODEL_NAME = "gpt4o"

# Graph configuration
GRAPH_OUTPUT_PATH = "output/knowledge_graph.gml"

# Text splitting configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 0


# Embedding configuration
EMBEDDING_MODEL_NAME = "text-embedding-qwen3-embedding-0.6b"
EMBEDDING_MODEL_URL = os.getenv("EMBEDDING_MODEL_URL")
