# LLM Memory Manager

This project implements a sophisticated memory and knowledge management system for Large Language Models (LLMs) using a graph-based approach. It transforms unstructured text data into a structured knowledge graph, enabling efficient and context-aware information retrieval.

## Features

- **Knowledge Graph Construction**: Automatically processes text documents, splits them into manageable chunks, and uses an LLM to identify and extract key entities (e.g., people, concepts, projects).
- **Graph-Based Storage**: Builds a `networkx` directed graph where text chunks and extracted entities are nodes, and their relationships (e.g., "mentions", "sequential") are edges.
- **Parallel Processing**: Leverages multi-threading to efficiently extract entities and generate embeddings for text chunks in parallel, significantly speeding up graph construction.
- **Embedding-Based Retrieval**: Generates vector embeddings for each text chunk and uses cosine similarity to find the most relevant information based on a user's query.
- **Contextual Expansion**: Goes beyond simple vector search by traversing the graph from the most relevant nodes to gather expanded context from neighboring chunks and entities.
- **Persistent Storage**: Saves and loads the constructed knowledge graph to/from a file, avoiding the need to re-process documents every time.
- **Configurable**: Easily configure API keys, model names, and other parameters through a `.env` file.

## How It Works

1.  **Data Ingestion**: The system reads a source text file.
2.  **Text Chunking**: The text is split into smaller, overlapping chunks.
3.  **Entity Extraction**: An LLM (e.g., GPT-4o) processes each chunk to extract key entities, which will become nodes in our graph.
4.  **Graph Building**: A `networkx` graph is created:
    -   Each text chunk becomes a `chunk` node.
    -   Each extracted entity becomes an `entity` node.
    -   Edges are created to link chunks to the entities they mention.
    -   Sequential chunks are linked to maintain document order.
5.  **Embedding Generation**: A text embedding model is used to generate a vector representation for each `chunk` node. These embeddings are cached on the node attributes.
6.  **Querying & Retrieval**:
    -   When a user provides a query, its embedding is generated.
    -   This query embedding is compared against all chunk embeddings to find the top `k` most similar chunks (Vector Search).
    -   The graph is then traversed outwards from these initial "seed" nodes to a specified depth, collecting all nodes in the subgraph.
    -   The content from all `chunk` nodes in this final, expanded set is aggregated to form a rich, relevant context.
7.  **Main Loop**: The `main.py` script orchestrates this process, allowing a user to repeatedly query the knowledge graph.

## Getting Started

### Prerequisites

-   Python 3.8+
-   An OpenAI-compatible API key and endpoint for both LLM completions and text embeddings.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/1nterstellar-JD/llm-memory-manager.git
    cd llm-memory-manager
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create a configuration file:**
    Create a file named `.env` in the root directory and add your API credentials. See the `.env.example` file for the required format.
    ```dotenv
    # .env
    OPENAI_API_KEY="your_api_key_here"
    OPENAI_BASE_URL="your_azure_endpoint_or_other_base_url"
    # Optional: If your embedding model is hosted elsewhere
    # EMBEDDING_MODEL_URL="your_embedding_endpoint"
    ```

### Usage

1.  **Add your data:**
    Place the text file you want to process into the `data/` directory. By default, the application looks for `data/sample.txt`.

2.  **Run the application:**
    ```bash
    python main.py
    ```

-   On the first run, the script will process the source text, build the knowledge graph, and save it to `output/knowledge_graph.gml`.
-   On subsequent runs, it will load the existing graph from the file, saving processing time.
-   You will then be prompted to enter queries in the terminal. Type `exit` to quit.

## Project Structure

```
/
├── data/                 # Input text files
│   └── sample.txt
├── output/               # Generated graph files
│   └── knowledge_graph.gml
├── src/                  # Source code
│   ├── config.py         # API clients, model names, env vars
│   ├── data_processor.py # Text loading and chunking
│   ├── graph_builder.py  # Entity extraction and graph construction
│   ├── graph_manager.py  # Saving and loading the graph
│   ├── logger.py         # Logging configuration
│   └── retriever.py      # Embedding generation and query retrieval logic
├── .env                  # (You create this) API keys and secrets
├── .gitignore
├── main.py               # Main application entry point
├── requirements.txt      # Project dependencies
└── README.md             # This file
```
