import asyncio
import json
from typing import List, Dict, Any, Tuple
import tiktoken

from src.config import llm_client, LLM_MODEL_NAME
from src.graph_db_manager import graph_db_manager
from src.logger import logger

# --- Tokenizer for Summarization ---
# Using a global tokenizer to avoid reloading it every time
try:
    tokenizer = tiktoken.encoding_for_model(LLM_MODEL_NAME)
except KeyError:
    logger.warning(f"Model {LLM_MODEL_NAME} not found for tiktoken. Falling back to cl100k_base.")
    tokenizer = tiktoken.get_encoding("cl100k_base")
# -----------------------------------

async def extract_entities_from_conversation(text: str) -> List[str]:
    """
    DEPRECATED: This function extracts entities but relationships are now preferred.
    Extracts key entities from a single piece of text (e.g., a user message or AI response).
    """
    # This is a simplified version. In a real scenario, we would use a more robust
    # implementation similar to the one in graph_builder.py
    prompt = f"""
    From the following text, extract key entities (people, places, concepts, projects).
    Respond with a JSON object with a single key 'entities' which is a list of strings.

    Text: "{text}"
    JSON Response:
    """
    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert at information extraction."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        entities = data.get("entities", [])
        # Normalize entities
        return sorted(list({e.lower().strip() for e in entities if isinstance(e, str) and e.strip()}))
    except Exception as e:
        logger.error(f"Failed to extract entities: {e}")
        return []

async def extract_relationships_from_conversation(text: str) -> List[Tuple[str, str, str]]:
    """
    Uses an LLM to extract knowledge triplets (subject, relation, object) from text.

    Args:
        text (str): The input text from the conversation.

    Returns:
        List[Tuple[str, str, str]]: A list of extracted triplets.
    """
    prompt = f"""
    From the text below, extract relationships between entities in the form of triplets (subject, relation, object).
    - The subject and object should be key entities.
    - The relation should be a short, descriptive verb phrase (e.g., 'is a', 'works for', 'located in').
    - Use present tense for relations where appropriate.
    - Normalize entity names to be consistent.

    Respond with a JSON object with a single key 'relationships' which is a list of lists,
    where each inner list is a [subject, relation, object] triplet.

    Example:
    Text: "Jules works at Acme Corp, a company based in New York."
    JSON Response: {{ "relationships": [["jules", "works at", "acme corp"], ["acme corp", "located in", "new york"]] }}

    Text to process:
    "{text}"

    JSON Response:
    """
    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert at knowledge graph extraction."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        # Basic validation
        raw_rels = data.get("relationships", [])
        validated_rels = []
        for rel in raw_rels:
            if isinstance(rel, list) and len(rel) == 3 and all(isinstance(i, str) for i in rel):
                 # Normalize and add
                validated_rels.append([item.lower().strip() for item in rel])
        return validated_rels
    except Exception as e:
        logger.error(f"Failed to extract relationships: {e}")
        return []

def store_triplets_in_graph(triplets: List[Tuple[str, str, str]]):
    """
    Stores a list of knowledge triplets in the Neo4j graph.
    Each triplet is used to create or merge two entity nodes and a relationship between them.

    Args:
        triplets (List[Tuple[str, str, str]]): The list of triplets to store.
    """
    if not triplets:
        return

    query = """
    UNWIND $triplets as triplet
    MERGE (subject:Entity {name: triplet[0]})
    MERGE (object:Entity {name: triplet[2]})
    // Replace special characters and spaces in relation for a valid type
    CALL apoc.create.relationship(subject, apoc.text.upperCamelCase(triplet[1]), {}, object)
    YIELD rel
    RETURN count(rel)
    """
    # Note: The above query uses APOC library for dynamic relationship types.
    # This is a powerful feature. A simpler version without APOC would be:
    # MERGE (subject)-[r:RELATED_TO {type: triplet[1]}]->(object)
    # However, dynamic types are generally better for this use case.
    # I will assume APOC is available on the user's Neo4j instance as it's standard.
    # If not, I will have to revise this.

    # Let's use a safer query that doesn't rely on APOC for now.
    # I can suggest installing it later as an optimization.

    safer_query = """
    UNWIND $triplets as triplet
    MERGE (subject:Entity {name: triplet[0]})
    MERGE (object:Entity {name: triplet[2]})
    // Storing relation type as a property is safer if APOC is not installed
    MERGE (subject)-[r:RELATED_TO {type: triplet[1]}]->(object)
    RETURN count(r)
    """

    try:
        result = graph_db_manager.execute_query(safer_query, parameters={"triplets": triplets})
        if result:
            count = result[0][0]
            logger.info(f"Stored {count} relationships in the graph.")
    except Exception as e:
        logger.error(f"Error storing triplets in graph: {e}")


async def process_conversation(messages: List[Dict[str, Any]]):
    """
    Orchestrates the processing of a conversation turn.
    It takes a list of messages, extracts relationships, and stores them in the graph.

    Args:
        messages (List[Dict[str, Any]]): A list of message dictionaries from the conversation.
    """
    logger.info(f"Processing a conversation with {len(messages)} messages.")

    full_text = " ".join([msg.get("content", "") for msg in messages if msg.get("content")])

    if not full_text.strip():
        logger.warning("Conversation contains no text content to process.")
        return

    # For simplicity, we extract from the whole text at once.
    # A more advanced approach could process message pairs.
    relationships = await extract_relationships_from_conversation(full_text)

    if relationships:
        logger.info(f"Extracted {len(relationships)} relationships from conversation.")
        store_triplets_in_graph(relationships)
    else:
        logger.info("No new relationships were extracted from the conversation.")

    logger.success("Conversation processing complete.")


def count_tokens_in_messages(messages: List[Dict[str, Any]]) -> int:
    """
    Calculates the total number of tokens for a list of messages using the pre-loaded tokenizer.

    Args:
        messages (List[Dict[str, Any]]): A list of message dictionaries.

    Returns:
        int: The total token count.
    """
    token_count = 0
    for message in messages:
        token_count += len(tokenizer.encode(message.get("content", "")))
    return token_count


async def summarize_conversation(messages: List[Dict[str, Any]]) -> str:
    """
    Generates a concise summary of a conversation history using an LLM.

    Args:
        messages (List[Dict[str, Any]]): The list of messages to summarize.

    Returns:
        str: The generated summary, or an error message.
    """
    logger.info("Summarizing conversation history...")
    conversation_text = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in messages
    )

    prompt = f"""Please provide a concise summary of the following conversation.
    The summary should capture the key topics, decisions, and important information exchanged.

    Conversation:
    {conversation_text}

    Summary:"""

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert at summarizing conversations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        summary = response.choices[0].message.content
        logger.success("Successfully generated conversation summary.")
        return summary
    except Exception as e:
        logger.error(f"Failed to summarize conversation: {e}")
        return "Error: Could not summarize the conversation."
