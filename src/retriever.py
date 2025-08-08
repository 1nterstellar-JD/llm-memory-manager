import networkx as nx
import numpy as np
from openai import OpenAI
from src.config import EMBEDDING_MODEL_NAME, OPENAI_API_KEY, EMBEDDING_MODEL_URL
from src.logger import logger

# Initialize a separate client for embeddings, as it might have a different base URL
# In your case, it points to the local service at http://127.0.0.1:1234/v1
# We will use the same client as in main.py for simplicity, assuming the base_url is set correctly.
client = OpenAI(api_key=OPENAI_API_KEY, base_url=EMBEDDING_MODEL_URL)


def get_embedding_from_api(text: str, model: str):
    """Generates embedding for a text using an OpenAI-compatible API."""
    try:
        # The API expects a list of strings, so we wrap the text in a list
        response = client.embeddings.create(input=[text], model=model)
        # Add a check to ensure data is not empty
        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        else:
            logger.error(
                f"API call successful, but no embedding data received. Response: {response}"
            )
            return None
    except Exception as e:
        logger.error(f"Error getting embedding from API: {e}")
        return None


def get_chunk_embeddings(G: nx.Graph):
    """Generates and caches embeddings for all chunk nodes in the graph via API call."""
    logger.info("Generating embeddings for graph chunks...")
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "chunk" and "embedding" not in data:
            content = data.get("content", "")
            embedding = get_embedding_from_api(content, model=EMBEDDING_MODEL_NAME)
            if embedding:
                G.nodes[node_id]["embedding"] = embedding
            else:
                logger.warning(f"Could not generate embedding for chunk {node_id}")


def retrieve_context(
    query: str, G: nx.Graph, top_k: int = 3, search_depth: int = 2
) -> str:
    """Retrieves relevant context from the graph for a given query."""
    if not G:
        return "Error: Knowledge graph not loaded."

    # 1. Generate embeddings for chunks if they don't exist
    get_chunk_embeddings(G)

    # 2. Find the most relevant chunk nodes using vector search
    query_embedding = get_embedding_from_api(query, model=EMBEDDING_MODEL_NAME)
    if query_embedding is None:
        return "Error: Could not generate embedding for the query."

    chunk_nodes = [
        (node_id, data)
        for node_id, data in G.nodes(data=True)
        if data.get("type") == "chunk" and "embedding" in data
    ]

    if not chunk_nodes:
        return "No text chunks found in the graph with embeddings."

    chunk_embeddings = np.array([data["embedding"] for _, data in chunk_nodes])
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Get top_k most similar chunk indices
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    initial_nodes = [chunk_nodes[i][0] for i in top_k_indices]

    # 3. Expand the context using graph traversal
    relevant_nodes = set(initial_nodes)
    for node in initial_nodes:
        # Use ego_graph to get all neighbors within a certain radius
        subgraph = nx.ego_graph(G, node, radius=search_depth)
        relevant_nodes.update(subgraph.nodes())

    # 4. Gather the content from the relevant nodes
    context_parts = []
    for node_id in relevant_nodes:
        if G.nodes[node_id].get("type") == "chunk":
            content = G.nodes[node_id].get("content", "")
            context_parts.append(content)

    # Deduplicate and join
    unique_context = "\n---\n".join(
        sorted(list(set(context_parts)), key=context_parts.index)
    )

    return unique_context


if __name__ == "__main__":
    vec = get_embedding_from_api("test", EMBEDDING_MODEL_NAME)
    print(vec)
