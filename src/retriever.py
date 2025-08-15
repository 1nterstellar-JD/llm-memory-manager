import asyncio
from src.config import embedding_client, EMBEDDING_MODEL_NAME
from src.logger import logger
from src.vector_store import conversation_vector_store

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

async def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Retrieves context purely from past conversations using semantic search.
    """
    logger.info("Retrieving context from past conversations...")
    
    query_embedding = await get_embedding_from_api(query, model=EMBEDDING_MODEL_NAME)
    if query_embedding is None:
        return "Error: Could not generate embedding for the query."

    # --- Semantic Search for Conversations ---
    conv_search_results = await conversation_vector_store.search(
        query_embedding, top_k, output_fields=["text"]
    )

    if not conv_search_results:
        logger.info("No relevant conversation snippets found.")
        return "" # Return empty string if no context is found

    conversation_context_parts = [hit['entity']['text'] for hit in conv_search_results if 'entity' in hit and 'text' in hit['entity']]

    if not conversation_context_parts:
        return ""

    # --- Format Context ---
    final_context = "Relevant past conversation snippets:\n" + "\n---\n".join(conversation_context_parts)
    logger.info("Added context from past conversations.")

    return final_context
