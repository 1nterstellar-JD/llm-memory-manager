# LLM Tool-Calling Agent with Long-Term Memory

[Read this document in Chinese (阅读中文文档)](README_zh.md)

This project provides an advanced framework for building AI agents that can use tools and have a persistent, long-term memory. It overcomes the standard limitations of finite context windows by implementing a sophisticated hybrid memory architecture, allowing the agent to remember past conversations, learn facts, and retrieve relevant context to make decisions.

The entire system is built with Python's `asyncio` to handle I/O-intensive operations like API calls and database interactions efficiently, ensuring a responsive user experience.

## Core Concept: Hybrid Memory

Standard LLMs are stateless. This engine solves that problem by integrating two powerful memory components that are automatically queried before every action:

1.  **Knowledge Graph (Neo4j):** For **structured, factual memory**. The system automatically extracts key entities and their relationships (e.g., "Jules works at Acme Inc.") from conversations and stores them as a graph. This allows for precise, fact-based lookups.
2.  **Vector Store (Milvus):** For **semantic, episodic memory**. It stores conversational turns as vector embeddings. This allows the system to find relevant past interactions based on their meaning and context, even if the wording is different.

By combining these two memories, the system automatically constructs a rich, comprehensive context to inform the agent's every decision.

## Features

-   **Persistent Long-Term Memory**: The agent remembers information across multiple sessions.
-   **Automatic Context Enrichment**: Before each turn, the agent's context is automatically enriched with relevant facts from its knowledge graph and past conversations from its vector store.
-   **Extensible Tool-Calling Framework**: Easily define and add new tools for the agent to perform external actions (e.g., fetching real-time data, calling APIs).
-   **Asynchronous Memory Processing**: Memory updates (relationship extraction, embedding) run as a background task, ensuring the agent is immediately ready for the next user input without blocking.
-   **Automatic Conversation Summarization**: To prevent context window overflow, the system automatically summarizes long conversations.
-   **Docker-Based Database Setup**: Includes a simple one-line command to set up a persistent Neo4j database with all necessary plugins.

## Architecture Overview

The system operates as a sophisticated agent loop:

1.  **User Input**: The user provides a prompt.
2.  **Automatic Memory Retrieval**: The `ContextManager` automatically queries the Neo4j graph and Milvus vector store to find relevant background information related to the user's query.
3.  **LLM Decides Action**: The user's query, the conversation history, and the retrieved background information are sent to the LLM. The LLM decides whether to answer directly or to use a tool.
4.  **Tool Execution (if needed)**: If the LLM decides to use a tool, the system executes the corresponding Python function and returns the result to the LLM.
5.  **Final Response**: The LLM uses the tool's output (or its initial reasoning) to generate a final, user-facing answer.
6.  **Asynchronous Memory Update**: The entire conversation turn (user query, tool calls, final answer) is processed in the background to extract new knowledge and store it in long-term memory for future use. The main loop does not wait for this to complete.

## Tech Stack

-   **Core**: Python 3.9+, asyncio
-   **LLM & Embeddings**: Any OpenAI-compatible API
-   **Knowledge Graph**: Neo4j
-   **Vector Store**: Milvus Lite (local file-based)
-   **Containerization**: Docker

## Setup and Installation

### Step 1: Prerequisites

-   Python 3.9+ 
-   Docker and Docker Compose

### Step 2: Clone and Set Up Environment

```bash
# Clone the repository
git clone https://github.com/your-username/llm-memory-manager.git
cd llm-memory-manager

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Launch Neo4j Database

Run the following command in your terminal to start a Neo4j container with the required APOC plugin. This command also ensures your data is persisted across restarts.

```bash
# Make sure to replace 'your_strong_password_here' with a secure password
docker run -d --name llm-memory-neo4j -p 7474:7474 -p 7687:7687 -v neo4j_data:/data --env NEO4J_AUTH="neo4j/your_strong_password_here" --env NEO4J_ACCEPT_LICENSE_AGREEMENT=yes --env NEO4J_PLUGINS='["apoc"]' neo4j:5.20.0-enterprise
```

### Step 4: Configure Application

Your application is configured via environment variables, which are read from `src/config.py`. For a production setup, using a `.env` file is recommended, but for now, you can edit the defaults in `src/config.py`.

1.  **Open `src/config.py`**.
2.  **Set your LLM provider credentials**: Update `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL_NAME`, and `EMBEDDING_MODEL_NAME`.
3.  **Set your Neo4j password**: Update `NEO4J_PASSWORD` to **exactly match the password** you used in the `docker run` command in Step 3.

## How to Run

Once the setup is complete, start the interactive agent with this command:

```bash
python main.py
```

The agent is now ready. Type `exit` to quit.

## How to Use: Example Interactions

Here are a few examples of how to interact with the agent:

-   **Using a Tool**:
    > `Your question: What time is it?`

-   **Adding a Fact to Memory**:
    > `Your question: My favorite project is called 'Project Phoenix'.`

-   **Retrieving a Fact from Memory**:
    > `Your question: What is the name of my favorite project?`