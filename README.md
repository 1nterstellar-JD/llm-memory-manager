# LLM Memory Manager: Asynchronous Knowledge Graph

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An asynchronous Python library for building and querying a sophisticated knowledge graph from unstructured text. It enhances Large Language Model (LLM) memory by creating a structured, context-aware knowledge base, enabling more intelligent and accurate information retrieval than simple vector search alone.

The entire system is built with `asyncio` to handle I/O-intensive operations like API calls and database interactions efficiently.

## Core Concepts

Standard LLM interactions are often limited by context window size and lack long-term memory. This project addresses that by transforming a linear stream of text (like conversation history or documents) into a rich knowledge graph.

This hybrid approach combines two powerful techniques:
1.  **Vector Search (Milvus):** For fast, semantic similarity search to find the most relevant starting points in the knowledge base.
2.  **Graph Traversal (NetworkX):** For expanding context by exploring connections and relationships from those starting points, uncovering related information that vector search might miss.

## Features

-   **Asynchronous from the Ground Up**: Built entirely with `asyncio` for high-performance, non-blocking I/O operations, perfect for integration into modern web services and applications.
-   **Hybrid Knowledge Graph**: Leverages a `networkx` graph for complex relationships and a Milvus vector store for efficient semantic search.
-   **Automated Entity Extraction**: Asynchronously calls an LLM to identify key entities (e.g., people, concepts, projects) from text chunks and builds them into the graph.
-   **Concurrent Processing**: Massively speeds up knowledge graph creation by processing entity extraction and embedding generation in parallel using `asyncio.gather`.
-   **Contextual Retrieval**: Goes beyond simple vector search by traversing the graph from the most relevant nodes to gather expanded, richer context.
-   **Persistent & Modular**: Saves the graph structure to a file and syncs embeddings with Milvus, with a modular design intended for use as a library.

## Architecture

The library is composed of several core modules:

-   `config.py`: Manages configuration, environment variables, and initializes asynchronous API clients.
-   `data_processor.py`: Handles loading and chunking of source text.
-   `graph_builder.py`: The asynchronous engine for constructing the `networkx` graph by extracting entities via LLM calls.
-   `vector_store.py`: An asynchronous wrapper for the Milvus client, managing vector storage and search.
-   `retriever.py`: The core query engine that performs the hybrid search: first hitting Milvus for top-k results, then traversing the graph for contextual expansion.
-   `graph_manager.py`: A simple utility for saving and loading the `networkx` graph structure.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/llm-memory-manager.git
    cd llm-memory-manager
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create a configuration file:**
    Create a file named `.env` in the root directory and add your API credentials.
    ```dotenv
    # .env
    OPENAI_API_KEY="your_api_key_here"
    OPENAI_BASE_URL="your_azure_endpoint_or_other_base_url"
    # Optional: If your embedding model is hosted elsewhere
    # EMBEDDING_MODEL_URL="your_embedding_endpoint"
    ```

## Library Usage Example

Here is a basic example of how to use the library in your own asynchronous application.

```python
import asyncio
from src.data_processor import load_and_split_text
from src.graph_builder import build_graph_from_chunks
from src.retriever import retrieve_context
from src.vector_store import vector_store
from src.graph_manager import save_graph, load_graph
from src.config import GRAPH_OUTPUT_PATH

async def run_example():
    # 1. Initialize the vector store
    await vector_store.setup()

    # 2. Load or build the graph
    knowledge_graph = load_graph(GRAPH_OUTPUT_PATH)
    if knowledge_graph is None:
        print("Building new knowledge graph...")
        chunks = load_and_split_text("data/sample.txt")
        if chunks:
            knowledge_graph = await build_graph_from_chunks(chunks)
            save_graph(knowledge_graph, GRAPH_OUTPUT_PATH)
            print("Graph built and saved.")
        else:
            print("No chunks created.")
            return

    # 3. Query the knowledge graph
    my_query = "What is the main concept discussed in the document?"
    print(f"\nQuerying for: '{my_query}'")
    
    context = await retrieve_context(my_query, knowledge_graph)
    
    print("\n--- Retrieved Context ---")
    print(context)
    print("--------------------------")

if __name__ == "__main__":
    asyncio.run(run_example())
```

## Example Application

This repository includes `main.py`, a fully functional interactive command-line application that demonstrates the library's capabilities.

To run it:
```bash
python main.py
```
-   On the first run, it will process the source text (`data/test.txt`), build the knowledge graph, and save it.
-   On subsequent runs, it will load the existing graph.
-   You will then be prompted to enter queries in the terminal. Type `exit` to quit.

## Project Structure

```
/
├── data/                 # Input text files
├── output/               # Saved graph file
├── milvus_data/          # Local Milvus Lite database
├── src/                  # Library source code
│   ├── config.py         # API clients and configuration
│   ├── data_processor.py # Text loading and chunking
│   ├── graph_builder.py  # Async graph construction
│   ├── graph_manager.py  # Graph saving/loading
│   ├── logger.py         # Logging configuration
│   ├── retriever.py      # Async hybrid retrieval logic
│   └── vector_store.py   # Async Milvus wrapper
├── .env                  # (You create this) API keys
├── .gitignore
├── main.py               # Example application entry point
├── requirements.txt      # Dependencies
└── README.md             # This file
```