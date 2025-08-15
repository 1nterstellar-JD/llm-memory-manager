# LLM Conversational Memory Engine

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project is an advanced conversational AI system that gives a Large Language Model (LLM) a persistent, long-term memory. It overcomes the standard limitations of finite context windows by implementing a sophisticated hybrid memory architecture, allowing the AI to remember past conversations, learn facts, and retrieve context with high accuracy.

The entire system is built with `asyncio` to handle I/O-intensive operations like API calls and database interactions efficiently.

## Core Concept: Hybrid Memory

Standard LLMs are stateless; they forget everything once a conversation ends. This engine solves that problem by integrating two powerful memory components:

1.  **Knowledge Graph (Neo4j):** For **structured, factual memory**. The system automatically extracts key entities and their relationships (e.g., "Bob works at Acme Inc.", "Acme Inc. is located in New York") from the conversation and stores them as a graph. This allows for precise queries and logical inference.
2.  **Vector Store (Milvus):** For **semantic, episodic memory**. It stores the raw conversational turns as vector embeddings. This allows the system to find relevant past interactions based on their meaning and context, even if the wording is different.

By combining these two memories, the system can retrieve both specific facts and relevant past snippets to form a rich, comprehensive context for the LLM's response.

## Features

-   **Persistent Long-Term Memory**: The AI remembers information across multiple sessions.
-   **Hybrid Context Retrieval**: Combines semantic search for conversational snippets and graph search for structured facts.
-   **Automated Knowledge Extraction**: In real-time, the system reads the conversation and automatically adds new facts to its knowledge graph.
-   **Relevance-Ranked Search**: Graph search results are ranked by relevance to provide the most accurate facts first.
-   **Automatic Conversation Summarization**: To prevent context window overflow, the system automatically summarizes long conversations.
-   **Interactive CLI**: A ready-to-use command-line interface for chatting with the memory-enabled AI.
-   **Fully Asynchronous**: Built with `asyncio` for high-performance, non-blocking I/O.

## Architecture Overview

```
                               +---------------------------+
                               |   LLM (e.g., OpenAI)      |
                               +-------------+-------------+
                                             ^
                                             | 3. Generate Answer
+--------------------------------------------+---------------------------------------------+
|                                    Prompt with Rich Context                              |
+------------------------------------------------------------------------------------------+
  ^                                       ^                                                ^
  | 2. Retrieve Context                   | 2. Retrieve Context                            |
+-------------------------+     +-------------------------+     +--------------------------+
| Knowledge Graph Facts   |     | Past Conversation       |     | Current User Query       |
| (from Neo4j)            |     | Snippets (from Milvus)  |     |                          |
+-------------------------+     +-------------------------+     +--------------------------+
  ^                                       ^
  | 1. Search Graph                       | 1. Semantic Search
+-------------------------+     +-------------------------+     +--------------------------+
| Knowledge Graph         |     | Vector Store            |     | User Query               |
| (Neo4j)                 |     | (Milvus)                |     | (e.g., "Who works here?")|
+-------------------------+     +-------------------------+     +--------------------------+
  | 5. Store Facts                        | 5. Store Turn                                  |
  +---------------------------------------+------------------------------------------------+
                                          | 4. Post-Process & Update Memory
                               +---------------------------+
                               | User Query + AI Answer    |
                               +---------------------------+
```

## Tech Stack

-   **Core**: Python 3.8+, asyncio
-   **LLM & Embeddings**: OpenAI
-   **Knowledge Graph**: Neo4j
-   **Vector Store**: Milvus
-   **Logging**: Loguru
-   **Configuration**: python-dotenv

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/llm-memory-manager.git
    cd llm-memory-manager
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Databases:**
    -   **Neo4j**: You must have a running Neo4j database instance.
    -   **Milvus**: The project uses Milvus Lite, which runs locally. Data will be stored in the `milvus_data/` directory. No separate setup is needed.

4.  **Create a configuration file:**
    Create a file named `.env` in the root directory and add your credentials.

    ```dotenv
    # .env

    # -- OpenAI API Credentials --
    OPENAI_API_KEY="your_openai_api_key"
    OPENAI_BASE_URL="your_openai_base_url_if_needed"
    LLM_MODEL_NAME="gpt-4"
    EMBEDDING_MODEL_NAME="text-embedding-ada-002"

    # -- Neo4j Database Connection --
    NEO4J_URI="bolt://localhost:7687"
    NEO4J_USER="neo4j"
    NEO4J_PASSWORD="your_neo4j_password"
    ```

## How to Run

Simply run the `main.py` script to start the interactive chat session.

```bash
python main.py
```

You can now start chatting with the AI. Type `exit` to quit. The AI will build its memory as you interact with it.

## Project Structure

```
/
├── data/                 # Sample text files for initial experiments
├── milvus_data/          # Local Milvus Lite database
├── src/                  # Library source code
│   ├── config.py         # API clients and configuration
│   ├── conversation_processor.py # Extracts facts and summarizes conversations
│   ├── graph_db_manager.py # Manages connection and queries to Neo4j
│   ├── logger.py         # Logging configuration
│   ├── retriever.py      # Hybrid retrieval logic (Graph + Vector)
│   └── vector_store.py   # Wrapper for the Milvus client
├── .env                  # (You create this) API keys and DB credentials
├── .gitignore
├── main.py               # Main application entry point
├── requirements.txt      # Dependencies
└── README.md             # This file
```
