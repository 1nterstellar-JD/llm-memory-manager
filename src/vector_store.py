
import os
import asyncio
from pymilvus.milvus_client import MilvusClient
from pymilvus import utility, Collection
from src.logger import logger
from src.config import MILVUS_URI, COLLECTION_NAME, VECTOR_DIMENSION

class VectorStore:
    def __init__(self):
        self._client = MilvusClient(uri=MILVUS_URI)
        self._collection_name = COLLECTION_NAME
        self.is_ready = asyncio.Event()

    async def setup(self):
        """Asynchronously initializes Milvus connection and sets up the collection."""
        try:
            if not await asyncio.to_thread(self._client.has_collection, self._collection_name):
                await self.acreate_collection()
            
            # The new MilvusClient manages collections differently, loading is often implicit
            logger.info(f"Successfully connected to Milvus and ensured collection '{self._collection_name}' exists.")
            self.is_ready.set()

        except Exception as e:
            logger.error(f"Failed to setup Milvus: {e}")
            self.is_ready.clear()
            raise

    async def acreate_collection(self):
        """Asynchronously creates the Milvus collection with a specific schema."""
        try:
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=VECTOR_DIMENSION,
                primary_field_name="id",
                vector_field_name="embedding",
                id_type="int",
                metric_type="L2"
            )
            logger.info(f"Collection '{self._collection_name}' created.")
            
            # Create index right after collection creation
            index_params = self._client.prepare_index_params(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="L2",
                params={"nlist": 128}
            )
            self._client.create_index(self._collection_name, index_params)
            logger.info("Index created for embedding field.")

        except Exception as e:
            logger.error(f"Failed to create collection or index: {e}")

    async def insert(self, node_id: int, vector: list):
        """Asynchronously inserts a vector into the collection."""
        await self.is_ready.wait()
        try:
            entities = [{"id": node_id, "embedding": vector}]
            await asyncio.to_thread(self._client.insert, collection_name=self._collection_name, data=entities)
            # Flush is handled automatically by the client in many cases, but can be called manually if needed
            # await asyncio.to_thread(self._client.flush, self._collection_name)
        except Exception as e:
            logger.error(f"Failed to insert vector for node {node_id}: {e}")

    async def search(self, query_vector: list, top_k: int) -> list:
        """Asynchronously searches for the most similar vectors."""
        await self.is_ready.wait()
        try:
            results = await asyncio.to_thread(
                self._client.search,
                collection_name=self._collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["id"]
            )
            return [hit['id'] for hit in results[0]]
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []

    async def count(self) -> int:
        """Asynchronously returns the number of vectors in the collection."""
        await self.is_ready.wait()
        try:
            # This is a synchronous call, but we await it to ensure setup is complete
            res = await asyncio.to_thread(self._client.query, self._collection_name, "id != 0", output_fields=["count(*)"])
            return res[0]['count(*)']
        except Exception as e:
            logger.error(f"Failed to count entities: {e}")
            return 0

    async def check_exists(self, node_ids: list[int]) -> list[int]:
        """Asynchronously checks which of the given node_ids exist in the collection."""
        await self.is_ready.wait()
        if not node_ids:
            return []
        try:
            ids_to_check = [int(i.split('_')[-1]) for i in node_ids]
            expr = f"id in {ids_to_check}"
            results = await asyncio.to_thread(self._client.query, self._collection_name, expr, output_fields=["id"])
            # The node_id in the graph is like 'chunk_123', but in milvus it's 123. We need to reconstruct the string.
            existing_ids_int = {res['id'] for res in results}
            return [f"chunk_{i}" for i in existing_ids_int]
        except Exception as e:
            logger.error(f"Failed to check existence for nodes: {e}")
            return []

# Singleton instance
vector_store = VectorStore()

async def main():
    # Example usage
    await vector_store.setup()
    logger.info(f"Number of vectors in store: {await vector_store.count()}")
    
    import numpy as np
    dummy_id = 1
    dummy_vec = list(np.random.rand(VECTOR_DIMENSION))
    
    await vector_store.insert(dummy_id, dummy_vec)
    await asyncio.sleep(1) # Give time for flush
    
    logger.info(f"Number of vectors in store after insert: {await vector_store.count()}")
    
    results = await vector_store.search(dummy_vec, top_k=1)
    logger.info(f"Search results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
