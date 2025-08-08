import networkx as nx
import os
from src.logger import logger

def save_graph(G: nx.Graph, file_path: str):
    """Saves a graph to a file in GML format."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        nx.write_gml(G, file_path)
        logger.success(f"Graph saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving graph: {e}")

def load_graph(file_path: str) -> nx.Graph | None:
    """Loads a graph from a GML file."""
    try:
        if os.path.exists(file_path):
            G = nx.read_gml(file_path)
            logger.success(f"Graph loaded successfully from {file_path}")
            return G
        else:
            logger.warning(f"Graph file not found at {file_path}. A new graph will be created.")
            return None
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        return None
