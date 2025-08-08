import networkx as nx
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Builds a knowledge graph from text chunks in parallel."""
    G = nx.DiGraph()
    
    # Create nodes for each chunk first
    for i, chunk in enumerate(chunks):
        G.add_node(f"chunk_{i}", type="chunk", content=chunk)

    # Use ThreadPoolExecutor to process chunks in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Map each chunk to a future
        future_to_chunk_index = {executor.submit(extract_entities_with_llm, chunk): i for i, chunk in enumerate(chunks)}
        
        for future in as_completed(future_to_chunk_index):
            chunk_index = future_to_chunk_index[future]
            try:
                entities = future.result()
                logger.info(f"Processed chunk {chunk_index + 1}/{len(chunks)} -> Found entities: {entities}")
                
                for entity in entities:
                    if not G.has_node(entity):
                        G.add_node(entity, type="entity")
                    G.add_edge(f"chunk_{chunk_index}", entity, type="mentions")

            except Exception as exc:
                logger.error(f"Chunk {chunk_index} generated an exception: {exc}")

    # Link sequential chunks (can be done after all nodes are processed)
    for i in range(len(chunks) - 1):
        G.add_edge(f"chunk_{i}", f"chunk_{i+1}", type="sequential")
            
    return G
