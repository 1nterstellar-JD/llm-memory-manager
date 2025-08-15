import numpy as np
import asyncio
import jieba
import json
from rank_bm25 import BM25Okapi

from src.config import embedding_client, EMBEDDING_MODEL_NAME
from src.logger import logger
from src.vector_store import doc_vector_store, conversation_vector_store
from src.graph_db_manager import graph_db_manager

# --- BM25 Retriever Cache ---
# We cache the model based on a graph state indicator (e.g., node count)
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

async def sync_embeddings_with_vector_store():
    """Ensures all chunk nodes in Neo4j have their embeddings in the vector store."""
    await doc_vector_store.is_ready.wait()
    logger.info("Syncing Neo4j chunks with the document vector store...")
    
    # Fetch all chunk nodes from Neo4j
    query = "MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id, c.content AS content"
    result = graph_db_manager.execute_query(query)
    if not result:
        logger.warning("Could not fetch chunks from Neo4j.")
        return

    all_chunk_nodes = {record["chunk_id"]: record["content"] for record in result}
    if not all_chunk_nodes:
        logger.info("No chunks in Neo4j to sync.")
        return

    # In the new vector store, IDs are integers.
    node_ids_to_check = []
    for node_id in all_chunk_nodes.keys():
        try:
            node_ids_to_check.append(int(node_id.split('_')[-1]))
        except (ValueError, IndexError):
            continue

    existing_ids = await doc_vector_store.check_exists(node_ids_to_check)
    
    insert_data = []
    tasks = []
    id_list_for_embedding = []

    for node_id, content in all_chunk_nodes.items():
        try:
            int_id = int(node_id.split('_')[-1])
            if int_id not in existing_ids:
                id_list_for_embedding.append(int_id)
                tasks.append(get_embedding_from_api(content, EMBEDDING_MODEL_NAME))
        except (ValueError, IndexError):
            continue

    if not tasks:
        logger.info("All chunk embeddings are already in the vector store.")
        return

    logger.info(f"Found {len(tasks)} chunks needing embedding.")
    embeddings = await asyncio.gather(*tasks)

    for int_id, embedding in zip(id_list_for_embedding, embeddings):
        if embedding:
            insert_data.append({"id": int_id, "embedding": embedding})
    
    if insert_data:
        await doc_vector_store.insert_batch(insert_data)
    logger.info("Sync with document vector store complete.")

def _initialize_bm25():
    """Initializes and caches a BM25 model using data from Neo4j."""
    query = "MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id, c.content AS content"
    result = graph_db_manager.execute_query(query)
    if not result:
        logger.warning("Could not fetch chunks from Neo4j for BM25.")
        return None

    chunk_docs = {record["chunk_id"]: list(jieba.cut(record["content"])) for record in result}
    if not chunk_docs:
        return None

    node_id_corpus = list(chunk_docs.keys())
    tokenized_corpus = [chunk_docs[node_id] for node_id in node_id_corpus]
    
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25 model initialized with data from Neo4j.")
    return bm25, node_id_corpus

async def retrieve_context(
    query: str, top_k: int = 3, search_depth: int = 1
) -> str:
    """Retrieves context using a hybrid search (Vector + Keyword) and graph traversal in Neo4j."""

    # --- 1. Hybrid Search for Documents ---
    query_embedding = await get_embedding_from_api(query, model=EMBEDDING_MODEL_NAME)
    if query_embedding is None:
        return "Error: Could not generate embedding for the query."

    # 1a. Vector Search for Documents
    doc_search_results = await doc_vector_store.search(query_embedding, top_k)
    doc_vector_ids = [hit['id'] for hit in doc_search_results]
    doc_vector_nodes = {f"chunk_{i}" for i in doc_vector_ids}
    logger.info(f"Document vector search found nodes: {doc_vector_nodes}")

    # 1b. Keyword Search for Documents
    bm25_result = _initialize_bm25()
    doc_keyword_nodes = set()
    if bm25_result:
        bm25, node_id_corpus = bm25_result
        tokenized_query = list(jieba.cut(query))
        doc_scores = bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(doc_scores)[::-1][:top_k]
        doc_keyword_nodes = {node_id_corpus[i] for i in top_n_indices}
        logger.info(f"Keyword search found nodes: {doc_keyword_nodes}")

    # --- 2. Graph Traversal for Document Context ---
    initial_nodes = doc_vector_nodes.union(doc_keyword_nodes)
    if not initial_nodes:
        return "No relevant chunks found for the query."
    logger.info(f"Combined initial-nodes: {initial_nodes}")

    # Use Cypher to find all nodes within the search_depth
    traversal_query = f"""
    MATCH (start:Chunk)
    WHERE start.chunk_id IN $start_nodes
    CALL {{
        WITH start
        MATCH (start)-[*0..{search_depth}]-(neighbor)
        RETURN neighbor
    }}
    RETURN DISTINCT neighbor
    """

    result = graph_db_manager.execute_query(traversal_query, parameters={"start_nodes": list(initial_nodes)})
    if not result:
        return "Could not retrieve context from graph."

    # --- 3. Semantic Search for Conversations ---
    conv_search_results = await conversation_vector_store.search(
        query_embedding, top_k, output_fields=["text"]
    )
    conversation_context_parts = [hit['entity']['text'] for hit in conv_search_results if 'text' in hit['entity']]
    
    # --- 4. Combine and Format Context ---
    final_context = ""
    if conversation_context_parts:
        conv_context = "\n---\n".join(conversation_context_parts)
        final_context += f"Relevant past conversation snippets:\n{conv_context}\n\n"
        logger.info("Added context from past conversations.")

    doc_context_parts = []
    if result:
        for record in result:
            node = record["neighbor"]
            # Ensure we only add content from Chunk nodes
            if "Chunk" in node.labels and "content" in node:
                doc_context_parts.append(node["content"])

    if doc_context_parts:
        unique_doc_context = "\n---\n".join(sorted(list(set(doc_context_parts)), key=doc_context_parts.index))
        final_context += f"Relevant information from documents:\n{unique_doc_context}"
        logger.info("Added context from documents.")

    if not final_context:
        return "No relevant information found in documents or past conversations."

    return final_context
