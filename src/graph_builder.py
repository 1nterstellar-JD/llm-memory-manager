import networkx as nx
from openai import AzureOpenAI
from src.config import OPENAI_API_KEY, LLM_MODEL_NAME, OPENAI_BASE_URL
from src.logger import logger

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    api_key=OPENAI_API_KEY,
    azure_endpoint=OPENAI_BASE_URL,
)


def extract_entities_with_llm(text_chunk: str):
    """Uses OpenAI to extract entities from a text chunk."""
    prompt = f"""Please extract the key entities (people, organizations, projects, concepts) from the following text.
    Respond with a JSON object containing a single key 'entities' which is a list of strings.
    Text: "{text_chunk}"

    JSON Response:"""

    try:
        response = client.chat.completions.create(
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
        # Correctly parsing the JSON string from the response
        import json

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
