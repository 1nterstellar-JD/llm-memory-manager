import json
import asyncio
from typing import List
from langchain_core.documents import Document
from src.config import llm_client, LLM_MODEL_NAME
from src.logger import logger
from src.graph_db_manager import graph_db_manager

# --- Concurrency & Retry Parameters ---
CONCURRENCY_LIMIT = 10
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
# -------------------------------------

async def extract_entities_with_llm(text_chunk: str, chunk_id: int, total_chunks: int):
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
                logger.info(f"Processed chunk {chunk_id + 1}/{total_chunks} -> Found entities: {entities}")
                return chunk_id, entities
            except Exception as e:
                if i < MAX_RETRIES - 1:
                    logger.warning(f"Chunk {chunk_id+1} failed with error: {e}. Retrying in {backoff:.1f}s...")
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error(f"Chunk {chunk_id+1} failed after {MAX_RETRIES} attempts. Final error: {e}")
                    return chunk_id, []
        
        return chunk_id, []

def setup_database_constraints():
    """Sets up unique constraints and indexes in the Neo4j database for optimization."""
    # Constraint for Chunk nodes
    graph_db_manager.execute_query(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
    )
    # Constraint for Entity nodes
    graph_db_manager.execute_query(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
    )
    logger.info("Database constraints and indexes are set up.")

async def build_graph_from_chunks(chunks: List[Document]):
    """Builds a knowledge graph in Neo4j from a list of Document objects asynchronously."""
    
    # First, create all chunk nodes
    for i, chunk in enumerate(chunks):
        query = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        ON CREATE SET c.content = $content, c.metadata = $metadata
        """
        parameters = {
            "chunk_id": f"chunk_{i}",
            "content": chunk.page_content,
            "metadata": json.dumps(chunk.metadata) # Store metadata as a JSON string
        }
        graph_db_manager.execute_query(query, parameters)
    logger.info(f"Successfully created {len(chunks)} chunk nodes in Neo4j.")

    # Second, extract entities and create entity nodes and relationships
    tasks = [
        extract_entities_with_llm(chunk.page_content, i, len(chunks))
        for i, chunk in enumerate(chunks)
    ]
    
    results = await asyncio.gather(*tasks)

    for chunk_index, raw_entities in results:
        if raw_entities:
            normalized_entities = {
                entity.lower().strip() for entity in raw_entities if isinstance(entity, str) and entity.lower().strip()
            }
            
            for entity_name in sorted(list(normalized_entities)):
                # Create entity node
                graph_db_manager.execute_query(
                    "MERGE (e:Entity {name: $name})", {"name": entity_name}
                )
                # Create relationship from chunk to entity
                query = """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MATCH (e:Entity {name: $entity_name})
                MERGE (c)-[:MENTIONS]->(e)
                """
                parameters = {"chunk_id": f"chunk_{chunk_index}", "entity_name": entity_name}
                graph_db_manager.execute_query(query, parameters)

    logger.info("Entity extraction and relationship creation complete.")

    # Third, create sequential relationships between chunks
    for i in range(len(chunks) - 1):
        query = """
        MATCH (c1:Chunk {chunk_id: $chunk_id_1})
        MATCH (c2:Chunk {chunk_id: $chunk_id_2})
        MERGE (c1)-[:SEQUENTIAL]->(c2)
        """
        parameters = {"chunk_id_1": f"chunk_{i}", "chunk_id_2": f"chunk_{i+1}"}
        graph_db_manager.execute_query(query, parameters)

    logger.info("Sequential relationships between chunks created.")
    return True # Indicate success
