import asyncio
from src.config import embedding_client, EMBEDDING_MODEL_NAME, llm_client, LLM_MODEL_NAME
from src.logger import logger
from src.vector_store import conversation_vector_store
from src.graph_db_manager import graph_db_manager
import json

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

async def extract_entities_from_query(query: str) -> list[str]:
    """Extracts key entities from the user's query using an LLM."""
    prompt = f"""
    From the following user question, please extract the key proper nouns, concepts, or entities.
    Focus on specific names of people, places, organizations, or projects.
    Respond with a JSON object containing a single key 'entities' which is a list of strings.
    If no specific entities are found, return an empty list.

    Question: "{query}"

    JSON Response:
    """
    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert at extracting key entities from text."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        entities = data.get("entities", [])
        # Normalize entities
        return sorted(list({e.lower().strip() for e in entities if isinstance(e, str) and e.strip()}))
    except Exception as e:
        logger.error(f"Failed to extract entities from query: {e}")
        return []

def query_graph_with_entities(entities: list[str]) -> str:
    """
    Queries the Neo4j graph for facts related to a list of entities.
    """
    if not entities:
        return ""
    
    all_facts = set()
    for entity in entities:
        # This query finds the entity and returns all its direct relationships and neighbors
        query = """
        MATCH (e:Entity {name: $entity_name})-[r]-(neighbor:Entity)
        RETURN e.name AS entity1, type(r) AS relation, neighbor.name AS entity2
        """
        results = graph_db_manager.execute_query(query, {"entity_name": entity})

        if results:
            for record in results:
                # To avoid duplicate relationships in different directions (e.g., A->B and B->A)
                # we can canonicalize the fact by sorting the entities alphabetically
                e1, e2 = sorted([record["entity1"], record["entity2"]])
                fact = f"({e1}) -[:{record['relation']}]-> ({e2})"
                all_facts.add(fact)

    if not all_facts:
        logger.info(f"No graph facts found for entities: {entities}")
        return ""

    facts_string = "\n".join(sorted(list(all_facts)))
    logger.info(f"Found {len(all_facts)} graph facts for entities: {entities}")
    return facts_string

async def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Retrieves context from memory using a hybrid approach:
    1. Semantic search for relevant conversation snippets.
    2. Graph search for structured facts related to entities in the query.
    """
    logger.info("Retrieving context from memory...")
    final_context_parts = []

    # --- 1. Semantic Search for Conversations ---
    query_embedding = await get_embedding_from_api(query, model=EMBEDDING_MODEL_NAME)
    if query_embedding:
        conv_search_results = await conversation_vector_store.search(
            query_embedding, top_k, output_fields=["text"]
        )
        if conv_search_results:
            conv_snippets = [hit['entity']['text'] for hit in conv_search_results if 'entity' in hit and 'text' in hit['entity']]
            if conv_snippets:
                final_context_parts.append("--- Relevant Past Conversation Snippets ---\n" + "\n".join(conv_snippets))
                logger.info("Retrieved context from conversation vector store.")

    # --- 2. Graph-based Search for Entities ---
    entities = await extract_entities_from_query(query)
    if entities:
        logger.info(f"Extracted entities from query: {entities}")
        graph_facts = query_graph_with_entities(entities)
        if graph_facts:
            final_context_parts.append("--- Knowledge Graph Facts ---\n" + graph_facts)
            logger.info("Retrieved context from knowledge graph.")

    # --- 3. Combine and Return ---
    if not final_context_parts:
        logger.info("No context found in any memory source.")
        return ""

    return "\n\n".join(final_context_parts)
