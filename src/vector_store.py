
import os
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from src.logger import logger
from src.config import MILVUS_URI, COLLECTION_NAME, VECTOR_DIMENSION

class VectorStore:
    def __init__(self):
        self.collection = None
        self.setup()

    def setup(self):
        """Initializes Milvus connection and sets up the collection."""
        try:
            # Use a local file for Milvus Lite
            if not os.path.exists(os.path.dirname(MILVUS_URI)):
                os.makedirs(os.path.dirname(MILVUS_URI))
            
            connections.connect("default", uri=MILVUS_URI)
            logger.info("Successfully connected to Milvus Lite.")

            if not utility.has_collection(COLLECTION_NAME):
                self.create_collection()
            
            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
            logger.info(f"Collection '{COLLECTION_NAME}' loaded.")

        except Exception as e:
            logger.error(f"Failed to setup Milvus: {e}")
            raise

    def create_collection(self):
        """Creates the Milvus collection with a specific schema."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        ]
        schema = CollectionSchema(fields, "Knowledge graph chunk embeddings")
        
        try:
            Collection(COLLECTION_NAME, schema)
            logger.info(f"Collection '{COLLECTION_NAME}' created.")
            self.create_index()
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")

    def create_index(self):
        """Creates an index for the embedding field for efficient search."""
        try:
            collection = Collection(COLLECTION_NAME)
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            collection.create_index("embedding", index_params)
            logger.info("Index created for embedding field.")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")

    def insert(self, node_id: int, vector: list):
        """Inserts a vector into the collection."""
        if not self.collection:
            logger.error("Collection not initialized.")
            return
        try:
            # Milvus expects lists of lists for entities
            entities = [
                {"id": node_id, "embedding": vector}
            ]
            self.collection.insert(entities)
            logger.info(f"Inserted vector for node {node_id}.")
        except Exception as e:
            logger.error(f"Failed to insert vector for node {node_id}: {e}")

    def search(self, query_vector: list, top_k: int) -> list:
        """Searches for the most similar vectors."""
        if not self.collection:
            logger.error("Collection not initialized.")
            return []
        try:
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["id"]
            )
            # Return a list of node IDs
            return [hit.id for hit in results[0]]
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []

    def count(self) -> int:
        """Returns the number of vectors in the collection."""
        if not self.collection:
            return 0
        return self.collection.num_entities

    def check_exists(self, node_ids: list) -> list:
        """Checks which of the given node_ids exist in the collection, and returns them."""
        if not self.collection or not node_ids:
            return []
        try:
            # Convert list to a string expression for the query
            expr = f"id in {node_ids}"
            results = self.collection.query(expr, output_fields=["id"])
            return [res['id'] for res in results]
        except Exception as e:
            logger.error(f"Failed to check existence for nodes: {e}")
            return []

# Singleton instance
vector_store = VectorStore()

if __name__ == "__main__":
    # Example usage
    logger.info(f"Number of vectors in store: {vector_store.count()}")
    # Dummy data
    import numpy as np
    dummy_id = 1
    dummy_vec = list(np.random.rand(VECTOR_DIMENSION))
    
    vector_store.insert(dummy_id, dummy_vec)
    vector_store.collection.flush() # Flush to make inserts searchable
    
    logger.info(f"Number of vectors in store after insert: {vector_store.count()}")
    
    results = vector_store.search(dummy_vec, top_k=1)
    logger.info(f"Search results: {results}")
