import networkx as nx
import json
from src.config import llm_client, LLM_MODEL_NAME
from src.logger import logger

def extract_entities_with_llm(text_chunk: str):
    """Uses OpenAI to extract entities from a text chunk."""
    prompt = f"""Please extract the key entities (people, organizations, projects, concepts) from the following text.
    Respond with a JSON object containing a single key 'entities' which is a list of strings.
    Text: "{text_chunk}"

    JSON Response:"""

    try:
        response = llm_client.chat.completions.create(
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
        return result.get("entities", [])
    except Exception as e:
        logger.error(f"An error occurred while calling OpenAI API: {e}")
        return []

def build_graph_from_chunks(chunks: list[str]):
    """Builds a knowledge graph from text chunks."""
    G = nx.DiGraph()

    # Create nodes for each chunk
    for i, chunk in enumerate(chunks):
        G.add_node(f"chunk_{i}", type="chunk", content=chunk)

    # Process each chunk to extract entities and create relationships
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
        entities = extract_entities_with_llm(chunk)

        for entity in entities:
            # Add entity node if it doesn't exist
            if not G.has_node(entity):
                G.add_node(entity, type="entity")

            # Add edge from chunk to entity
            G.add_edge(f"chunk_{i}", entity, type="mentions")

        # Link sequential chunks
        if i > 0:
            G.add_edge(f"chunk_{i-1}", f"chunk_{i}", type="sequential")

    return G