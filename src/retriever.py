import networkx as nx
import numpy as np
import asyncio
from src.config import embedding_client, EMBEDDING_MODEL_NAME
from src.logger import logger
from src.vector_store import vector_store

async def get_embedding_from_api(text: str, model: str):
    """Asynchronously generates embedding for a text using an OpenAI-compatible API."""
    try:
        response = await embedding_client.embeddings.create(input=[text], model=model)
        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        else:
            logger.error(f"API call successful, but no embedding data received. Response: {response}")
            return None
    except Exception as e:
        logger.error(f"Error getting embedding from API: {e}")
        return None

async def sync_embeddings_with_vector_store(G: nx.Graph):
    """
    Asynchronously ensures that all chunk nodes in the graph have their embeddings in Milvus.
    """
    await vector_store.is_ready.wait()
    logger.info("Syncing graph chunks with the vector store...")
    
    all_chunk_nodes = {
        node_id: data.get('content', '')
        for node_id, data in G.nodes(data=True)
        if data.get('type') == 'chunk'
    }

    if not all_chunk_nodes:
        logger.info("No chunk nodes found in the graph.")
        return

    node_ids_list = list(all_chunk_nodes.keys())
    existing_ids = await vector_store.check_exists(node_ids_list)
    
    chunks_to_process = {
        node_id: content 
        for node_id, content in all_chunk_nodes.items() 
        if node_id not in existing_ids
    }

    if not chunks_to_process:
        logger.info("All chunk embeddings are already in the vector store.")
        return

    logger.info(f"Found {len(chunks_to_process)} chunks needing embedding.")

    tasks = [
        get_embedding_from_api(content, EMBEDDING_MODEL_NAME)
        for content in chunks_to_process.values()
    ]
    node_ids_for_tasks = list(chunks_to_process.keys())

    embeddings = await asyncio.gather(*tasks)

    insert_tasks = []
    for node_id, embedding in zip(node_ids_for_tasks, embeddings):
        if embedding:
            # Extract integer from node_id like 'chunk_123'
            int_id = int(node_id.split('_')[-1])
            insert_tasks.append(vector_store.insert(int_id, embedding))
        else:
            logger.warning(f"Failed to generate embedding for chunk {node_id}")
    
    await asyncio.gather(*insert_tasks)
    logger.info("Sync with vector store complete.")

async def retrieve_context(
    query: str, G: nx.Graph, top_k: int = 3, search_depth: int = 2
) -> str:
    """Asynchronously retrieves relevant context from the graph for a given query."""
    if not G or G.number_of_nodes() == 0:
        return "Error: Knowledge graph not loaded or is empty."

    await sync_embeddings_with_vector_store(G)

    query_embedding = await get_embedding_from_api(query, model=EMBEDDING_MODEL_NAME)
    if query_embedding is None:
        return "Error: Could not generate embedding for the query."

    logger.info("Searching for relevant nodes in vector store...")
    initial_node_ids = await vector_store.search(query_embedding, top_k)
    initial_nodes = [f"chunk_{i}" for i in initial_node_ids]

    if not initial_nodes:
        return "No relevant chunks found in the vector store for the query."
    
    logger.info(f"Found initial nodes: {initial_nodes}")

    relevant_nodes = set(initial_nodes)
    for node in initial_nodes:
        if node not in G:
            logger.warning(f"Node {node} found by vector search but not in graph. Skipping.")
            continue
        subgraph = nx.ego_graph(G, node, radius=search_depth)
        relevant_nodes.update(subgraph.nodes())

    context_parts = []
    for node_id in relevant_nodes:
        if G.nodes[node_id].get("type") == "chunk":
            content = G.nodes[node_id].get("content", "")
            context_parts.append(content)

    unique_context = "\n---\n".join(
        sorted(list(set(context_parts)), key=context_parts.index)
    )

    return unique_context

