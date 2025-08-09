import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import embedding_client, EMBEDDING_MODEL_NAME
from src.logger import logger
from src.vector_store import vector_store

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

def sync_embeddings_with_vector_store(G: nx.Graph):
    """
    Ensures that all chunk nodes in the graph have their embeddings in Milvus.
    This function is now decoupled from storing embeddings in the graph itself.
    """
    logger.info("Syncing graph chunks with the vector store...")
    
    all_chunk_nodes = {
        node_id: data.get('content', '')
        for node_id, data in G.nodes(data=True)
        if data.get('type') == 'chunk'
    }

    if not all_chunk_nodes:
        logger.info("No chunk nodes found in the graph.")
        return

    # Check which nodes are already in the vector store
    node_ids_list = list(all_chunk_nodes.keys())
    existing_ids = set(vector_store.check_exists(node_ids_list))
    
    chunks_to_process = {
        node_id: content 
        for node_id, content in all_chunk_nodes.items() 
        if node_id not in existing_ids
    }

    if not chunks_to_process:
        logger.info("All chunk embeddings are already in the vector store.")
        return

    logger.info(f"Found {len(chunks_to_process)} chunks needing embedding.")

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
                    # Insert into Milvus, not the graph node
                    vector_store.insert(node_id, embedding)
                else:
                    logger.warning(f"Failed to generate embedding for chunk {node_id}")
            except Exception as exc:
                logger.error(f"Chunk {node_id} generated an exception during embedding: {exc}")
    
    # Flush inserts to make them searchable
    vector_store.collection.flush()
    logger.info("Sync with vector store complete.")


def retrieve_context(
    query: str, G: nx.Graph, top_k: int = 3, search_depth: int = 2
) -> str:
    """Retrieves relevant context from the graph for a given query using Milvus."""
    if not G or G.number_of_nodes() == 0:
        return "Error: Knowledge graph not loaded or is empty."

    # 1. Ensure embeddings are in the vector store
    sync_embeddings_with_vector_store(G)

    # 2. Find the most relevant chunk nodes using vector search in Milvus
    query_embedding = get_embedding_from_api(query, model=EMBEDDING_MODEL_NAME)
    if query_embedding is None:
        return "Error: Could not generate embedding for the query."

    logger.info("Searching for relevant nodes in vector store...")
    initial_nodes = vector_store.search(query_embedding, top_k)

    if not initial_nodes:
        return "No relevant chunks found in the vector store for the query."
    
    logger.info(f"Found initial nodes: {initial_nodes}")

    # 3. Expand the context using graph traversal
    relevant_nodes = set(initial_nodes)
    for node in initial_nodes:
        if node not in G:
            logger.warning(f"Node {node} found by vector search but not in graph. Skipping.")
            continue
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
    # Example of getting an embedding.
    # The main logic would now require a graph to be loaded.
    vec = get_embedding_from_api("test", EMBEDDING_MODEL_NAME)
    print(f"Test embedding vector (first 10 dims): {vec[:10] if vec else 'None'}")
