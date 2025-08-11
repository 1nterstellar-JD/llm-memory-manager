import networkx as nx
import json
import asyncio
from typing import List
from langchain_core.documents import Document
from src.config import llm_client, LLM_MODEL_NAME
from src.logger import logger

# --- Concurrency & Retry Parameters ---
CONCURRENCY_LIMIT = 10
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
# -------------------------------------

async def extract_entities_with_llm(text_chunk: str, chunk_index: int, total_chunks: int):
    """Asynchronously uses OpenAI to extract entities, with concurrency limiting and retry logic."""
    async with semaphore:
        prompt = f"""Please extract the key entities (people, organizations, projects, concepts) from the following text.
        Respond with a JSON object containing a single key 'entities' which is a list of strings.
        Text: "{text_chunk}"

        JSON Response:"""
        
        backoff = INITIAL_BACKOFF
        for i in range(MAX_RETRIES):
            try:
                response = await llm_client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant designed to extract information and return it in JSON format.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                result = json.loads(response.choices[0].message.content)
                entities = result.get("entities", [])
                logger.info(f"Processed chunk {chunk_index + 1}/{total_chunks} -> Found entities: {entities}")
                return chunk_index, entities  # Success
            except Exception as e:
                if i < MAX_RETRIES - 1:
                    logger.warning(f"Chunk {chunk_index+1} failed with error: {e}. Retrying in {backoff:.1f}s...")
                    await asyncio.sleep(backoff)
                    backoff *= 2  # Exponential backoff
                else:
                    logger.error(f"Chunk {chunk_index+1} failed after {MAX_RETRIES} attempts. Final error: {e}")
                    return chunk_index, []  # All retries failed
        
        return chunk_index, [] # Should not be reached, but as a fallback

async def build_graph_from_chunks(chunks: List[Document]):
    """Builds a knowledge graph from a list of Document objects asynchronously."""
    G = nx.DiGraph()
    
    for i, chunk in enumerate(chunks):
        G.add_node(f"chunk_{i}", type="chunk", content=chunk.page_content, metadata=chunk.metadata)

    tasks = [
        extract_entities_with_llm(chunk.page_content, i, len(chunks))
        for i, chunk in enumerate(chunks)
    ]
    
    results = await asyncio.gather(*tasks)

    for chunk_index, raw_entities in results:
        if raw_entities:
            # Normalize and filter entities
            normalized_entities = set()
            for entity in raw_entities:
                if isinstance(entity, str):
                    # Normalize: lowercase and strip whitespace
                    normalized = entity.lower().strip()
                    if normalized: # Ensure not empty after stripping
                        normalized_entities.add(normalized)
                else:
                    logger.warning(f"Ignoring non-string entity '{entity}' in chunk {chunk_index+1}")
            
            # Add normalized entities to the graph
            for entity in sorted(list(normalized_entities)):
                if not G.has_node(entity):
                    G.add_node(entity, type="entity")
                G.add_edge(f"chunk_{chunk_index}", entity, type="mentions")

    for i in range(len(chunks) - 1):
        G.add_edge(f"chunk_{i}", f"chunk_{i+1}", type="sequential")
            
    return G
