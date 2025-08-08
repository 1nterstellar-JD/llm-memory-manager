import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import embedding_client, EMBEDDING_MODEL_NAME
from src.logger import logger

def get_embedding_from_api(text: str, model: str):
    """Generates embedding for a text using an OpenAI-compatible API."""
    try:
        response = embedding_client.embeddings.create(input=[text], model=model)
        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        else:
            logger.error(f"API call successful, but no embedding data received. Response: {response}")
            return None
    except Exception as e:
        logger.error(f"Error getting embedding from API: {e}")
        return None

def get_chunk_embeddings(G: nx.Graph):
    """Generates and caches embeddings for all chunk nodes in parallel."""
    logger.info("Generating embeddings for graph chunks...")
    
    chunks_to_process = {
        node_id: data.get('content', '')
        for node_id, data in G.nodes(data=True)
        if data.get('type') == 'chunk' and 'embedding' not in data
    }

    if not chunks_to_process:
        logger.info("All chunk embeddings are already cached.")
        return

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_node_id = {
            executor.submit(get_embedding_from_api, content, EMBEDDING_MODEL_NAME): node_id
            for node_id, content in chunks_to_process.items()
        }

        for future in as_completed(future_to_node_id):
            node_id = future_to_node_id[future]
            try:
                embedding = future.result()
                if embedding:
                    G.nodes[node_id]['embedding'] = embedding
                    logger.info(f"Generated embedding for chunk {node_id}")
                else:
                    logger.warning(f"Failed to generate embedding for chunk {node_id}")
            except Exception as exc:
                logger.error(f"Chunk {node_id} generated an exception during embedding: {exc}")

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