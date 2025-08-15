
import asyncio
from src.graph_db_manager import graph_db_manager
from src.logger import logger

async def clear_database():
    """Connects to Neo4j and deletes all nodes and relationships."""
    if not graph_db_manager:
        logger.error("GraphDBManager is not initialized. Cannot clear database.")
        return

    try:
        logger.info("Connecting to Neo4j to clear the database...")
        
        # The connection is already established by the singleton instance.
        # We just need to execute the query.
        
        query = "MATCH (n) DETACH DELETE n"
        logger.info(f"Executing query: {query}")
        
        result = graph_db_manager.execute_query(query)
        
        # To get the count of deleted nodes, a different query would be needed
        # like "MATCH (n) DETACH DELETE n RETURN count(n)".
        # For simplicity, we just confirm execution.
        
        # The execute_query returns a list of records or None.
        # A successful DETACH DELETE returns an empty list.
        if result is not None:
            # We can get more detailed summary if needed, but for now, this is enough.
            logger.success("Successfully cleared the Neo4j database.")
        else:
            logger.error("Failed to clear the database. The query execution returned None.")

    except Exception as e:
        logger.error(f"An error occurred while trying to clear the database: {e}")
    finally:
        # Close the connection
        if graph_db_manager:
            graph_db_manager.close()
        logger.info("Neo4j connection closed.")

if __name__ == "__main__":
    asyncio.run(clear_database())
