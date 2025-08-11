import networkx as nx
import numpy as np
import asyncio
import jieba
from rank_bm25 import BM25Okapi

from src.config import embedding_client, EMBEDDING_MODEL_NAME
from src.logger import logger
from src.vector_store import vector_store

# --- BM25 Retriever Cache ---
_bm25_cache = {}
# ---------------------------

async def get_embedding_from_api(text: str, model: str):
    """Asynchronously generates embedding for a text using an OpenAI-compatible API."""
    try:
        response = await embedding_client.embeddings.create(input=[text], model=model)
        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        return None
    except Exception as e:
        logger.error(f"Error getting embedding from API: {e}")
        return None

async def sync_embeddings_with_vector_store(G: nx.Graph):
    """Ensures all chunk nodes in the graph have their embeddings in the vector store."""
    await vector_store.is_ready.wait()
    logger.info("Syncing graph chunks with the vector store...")
    
    all_chunk_nodes = {
        node_id: data.get('content', '')
        for node_id, data in G.nodes(data=True)
        if data.get('type') == 'chunk'
    }

    if not all_chunk_nodes:
        return

    node_ids_list = list(all_chunk_nodes.keys())
    existing_ids = await vector_store.check_exists(node_ids_list)
    
    chunks_to_process = {node_id: content for node_id, content in all_chunk_nodes.items() if node_id not in existing_ids}

    if not chunks_to_process:
        logger.info("All chunk embeddings are already in the vector store.")
        return

    logger.info(f"Found {len(chunks_to_process)} chunks needing embedding.")
    tasks = [get_embedding_from_api(content, EMBEDDING_MODEL_NAME) for content in chunks_to_process.values()]
    embeddings = await asyncio.gather(*tasks)

    insert_data = []
    for node_id, embedding in zip(chunks_to_process.keys(), embeddings):
        if embedding:
            int_id = int(node_id.split('_')[-1])
            insert_data.append((int_id, embedding))
    
    if insert_data:
        await vector_store.insert_batch(insert_data)
    logger.info("Sync with vector store complete.")

def _initialize_bm25(G: nx.Graph):
    """Initializes and caches a BM25 model using jieba for Chinese tokenization."""
    graph_id = id(G)
    if graph_id in _bm25_cache:
        return _bm25_cache[graph_id]

    logger.info("Initializing BM25 model with jieba for keyword search...")
    chunk_docs = {
        node_id: list(jieba.cut(data.get('content', '')))
        for node_id, data in G.nodes(data=True)
        if data.get('type') == 'chunk'
    }
    
    if not chunk_docs:
        return None

    node_id_corpus = list(chunk_docs.keys())
    tokenized_corpus = [chunk_docs[node_id] for node_id in node_id_corpus]
    
    bm25 = BM25Okapi(tokenized_corpus)
    _bm25_cache[graph_id] = (bm25, node_id_corpus)
    logger.info("BM25 model initialized.")
    return bm25, node_id_corpus

async def retrieve_context(
    query: str, G: nx.Graph, top_k: int = 3, search_depth: int = 1
) -> str:
    """Retrieves context using a hybrid search (Vector + Keyword with jieba) and graph traversal."""
    if not G or G.number_of_nodes() == 0:
        return "Error: Knowledge graph not loaded or is empty."

    await sync_embeddings_with_vector_store(G)

    # --- 1. Vector Search ---
    query_embedding = await get_embedding_from_api(query, model=EMBEDDING_MODEL_NAME)
    if query_embedding is None:
        return "Error: Could not generate embedding for the query."
    vector_search_ids = await vector_store.search(query_embedding, top_k)
    vector_nodes = {f"chunk_{i}" for i in vector_search_ids}
    logger.info(f"Vector search found nodes: {vector_nodes}")

    # --- 2. Keyword Search (BM25 with jieba) ---
    bm25_result = _initialize_bm25(G)
    keyword_nodes = set()
    if bm25_result:
        bm25, node_id_corpus = bm25_result
        tokenized_query = list(jieba.cut(query))
        doc_scores = bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(doc_scores)[::-1][:top_k]
        keyword_nodes = {node_id_corpus[i] for i in top_n_indices}
        logger.info(f"Keyword search found nodes: {keyword_nodes}")

    # --- 3. Combine and Expand ---
    initial_nodes = vector_nodes.union(keyword_nodes)
    if not initial_nodes:
        return "No relevant chunks found for the query."
    logger.info(f"Combined initial-nodes: {initial_nodes}")

    relevant_nodes = set(initial_nodes)
    for node in initial_nodes:
        if node in G:
            subgraph = nx.ego_graph(G, node, radius=search_depth)
            relevant_nodes.update(subgraph.nodes())
        else:
            logger.warning(f"Node {node} found by search but not in graph. Skipping.")

    # --- 4. Format Context ---
    context_parts = []
    for node_id in relevant_nodes:
        if G.nodes[node_id].get("type") == "chunk":
            context_parts.append(G.nodes[node_id].get("content", ""))

    unique_context = "\n---\n".join(sorted(list(set(context_parts)), key=context_parts.index))
    return unique_context

if __name__ == "__main__":
    # This is a focused test block to verify the get_embedding_from_api function.
    # You can run this file directly `python src/retriever.py` to test.
    
    async def test_embedding_api():
        logger.info("--- Running get_embedding_from_api Test ---")
        
        test_text = "Hello, world!"
        model_name = EMBEDDING_MODEL_NAME
        
        logger.info(f"Requesting embedding for: '{test_text}' with model '{model_name}'")
        
        embedding = await get_embedding_from_api(test_text, model_name)
        
        print("\n--- Test Results ---")
        if embedding:
            print(f"Successfully retrieved embedding!")
            print(f"Vector dimension: {len(embedding)}")
            print(f"First 5 dimensions: {embedding[:5]}")
        else:
            print("Failed to retrieve embedding. Check logs for errors.")
        print("--- End of Test ---")

        # Clean up the client connection
        await embedding_client.close()

    # Execute the test
    try:
        asyncio.run(test_embedding_api())
    except Exception as e:
        logger.error(f"An error occurred during the test run: {e}")