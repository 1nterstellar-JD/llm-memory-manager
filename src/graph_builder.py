import networkx as nx
import json
import asyncio
from typing import List
from langchain_core.documents import Document
from src.config import llm_client, LLM_MODEL_NAME
from src.logger import logger

async def extract_entities_with_llm(text_chunk: str, chunk_index: int, total_chunks: int):
    """Asynchronously uses OpenAI to extract entities from a text chunk."""
    prompt = f"""Please extract the key entities (people, organizations, projects, concepts) from the following text.
    Respond with a JSON object containing a single key 'entities' which is a list of strings.
    Text: "{text_chunk}"

    JSON Response:"""

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
        return chunk_index, entities
    except Exception as e:
        logger.error(f"An error occurred in chunk {chunk_index} while calling OpenAI API: {e}")
        return chunk_index, []

async def build_graph_from_chunks(chunks: List[Document]):
    """Builds a knowledge graph from a list of Document objects asynchronously."""
    G = nx.DiGraph()
    
    # Create nodes for each chunk first
    for i, chunk in enumerate(chunks):
        G.add_node(f"chunk_{i}", type="chunk", content=chunk.page_content, metadata=chunk.metadata)

    # Create concurrent tasks for entity extraction
    tasks = [
        extract_entities_with_llm(chunk.page_content, i, len(chunks))
        for i, chunk in enumerate(chunks)
    ]
    
    results = await asyncio.gather(*tasks)

    # Process results and build the graph
    for chunk_index, entities in results:
        for entity in entities:
            if not G.has_node(entity):
                G.add_node(entity, type="entity")
            G.add_edge(f"chunk_{chunk_index}", entity, type="mentions")

    # Link sequential chunks
    for i in range(len(chunks) - 1):
        G.add_edge(f"chunk_{i}", f"chunk_{i+1}", type="sequential")
            
    return G
