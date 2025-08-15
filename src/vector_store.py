import os
import asyncio
from pymilvus.milvus_client import MilvusClient
from src.logger import logger
from src.config import (
    MILVUS_URI,
    CONVERSATION_COLLECTION_NAME,
    VECTOR_DIMENSION
)

class VectorCollectionManager:
    def __init__(self, collection_name: str, id_type: str = "int", id_max_length: int = 65535, dynamic_field: bool = False):
        self._collection_name = collection_name
        self._id_type = id_type
        self._id_max_length = id_max_length
        self._enable_dynamic_field = dynamic_field
        # Note: The client is shared across all instances
        if not hasattr(VectorCollectionManager, '_client'):
            db_dir = os.path.dirname(MILVUS_URI)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"Created Milvus data directory: {db_dir}")
            VectorCollectionManager._client = MilvusClient(uri=MILVUS_URI)

        self.is_ready = asyncio.Event()

    async def setup(self):
        """Initializes Milvus connection and sets up the collection."""
        try:
            if not await asyncio.to_thread(self._client.has_collection, self._collection_name):
                await self._create_collection()
            
            logger.info(f"Successfully connected to Milvus and ensured collection '{self._collection_name}' exists.")
            self.is_ready.set()
        except Exception as e:
            logger.error(f"Failed to setup Milvus collection '{self._collection_name}': {e}")
            self.is_ready.clear()
            raise

    async def _create_collection(self):
        """Creates the Milvus collection with a specific schema."""
        try:
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=VECTOR_DIMENSION,
                primary_field_name="id",
                vector_field_name="embedding",
                id_type=self._id_type,
                max_length=self._id_max_length,
                metric_type="L2",
                enable_dynamic_field=self._enable_dynamic_field
            )
            logger.info(f"Collection '{self._collection_name}' created (dynamic fields: {self._enable_dynamic_field}).")
            
            index_params = self._client.prepare_index_params(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="L2",
                params={"nlist": 128}
            )
            self._client.create_index(self._collection_name, index_params)
            logger.info(f"Index created for '{self._collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to create collection or index for '{self._collection_name}': {e}")

    async def insert_batch(self, data: list):
        """Inserts a batch of vectors. Data format depends on id_type."""
        await self.is_ready.wait()
        if not data:
            return
        try:
            # Data is expected to be a list of dicts: [{"id": ..., "embedding": ...}]
            await asyncio.to_thread(self._client.insert, collection_name=self._collection_name, data=data)
        except Exception as e:
            logger.error(f"Failed to batch insert into '{self._collection_name}': {e}")

    async def search(self, query_vector: list, top_k: int, output_fields: list = ["id"]) -> list:
        """Searches for the most similar vectors."""
        await self.is_ready.wait()
        try:
            results = await asyncio.to_thread(
                self._client.search,
                collection_name=self._collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=output_fields
            )
            return results[0]  # Return list of hits
        except Exception as e:
            logger.error(f"Failed to search in '{self._collection_name}': {e}")
            return []

    async def check_exists(self, ids: list) -> set:
        """Checks which of the given ids exist in the collection."""
        await self.is_ready.wait()
        if not ids:
            return set()
        try:
            expr = f"id in {ids}"
            results = await asyncio.to_thread(self._client.query, self._collection_name, expr, output_fields=["id"])
            return {res['id'] for res in results}
        except Exception as e:
            logger.error(f"Failed to check existence in '{self._collection_name}': {e}")
            return set()

    @classmethod
    async def close_all(cls):
        """Closes the shared Milvus client connection."""
        if hasattr(cls, '_client'):
            logger.info("Closing shared Milvus client connection.")
            await asyncio.to_thread(cls._client.close)

# Singleton instance for the conversation collection
conversation_vector_store = VectorCollectionManager(
    CONVERSATION_COLLECTION_NAME,
    id_type="string",
    id_max_length=1000, # A timestamp-based ID won't be this long
    dynamic_field=True
)
