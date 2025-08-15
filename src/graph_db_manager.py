from neo4j import GraphDatabase, Driver
from src.logger import logger
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class GraphDBManager:
    """A singleton class to manage the connection to the Neo4j database."""
    _driver: Driver = None

    def __init__(self):
        """
        The constructor for GraphDBManager class.
        It is designed to be a singleton, so this should not be called directly.
        """
        # This is a singleton, so the constructor should not be called more than once.
        if GraphDBManager._driver is not None:
            raise Exception("This class is a singleton! Use the get_instance method.")

        try:
            # Establish the connection driver
            GraphDBManager._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            # Verify that the connection is valid
            GraphDBManager._driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            # Reraise the exception to be handled by the application's entry point
            raise

    def close(self):
        """Closes the database connection driver."""
        if self._driver is not None:
            self._driver.close()
            logger.info("Neo4j connection closed.")

    def execute_query(self, query: str, parameters: dict = None):
        """
        Executes a Cypher query against the database.

        Args:
            query (str): The Cypher query to execute.
            parameters (dict, optional): A dictionary of parameters to pass to the query. Defaults to None.

        Returns:
            A list of records if the query is successful, None otherwise.
        """
        # Ensure the driver is initialized
        if self._driver is None:
            logger.error("Driver not initialized. Cannot execute query.")
            return None

        with self._driver.session() as session:
            try:
                result = session.run(query, parameters)
                # Eagerly consume the result to catch potential errors
                return [record for record in result]
            except Exception as e:
                logger.error(f"Error executing Cypher query: {e}")
                return None

# Singleton instance of the manager
try:
    graph_db_manager = GraphDBManager()
except Exception:
    # If connection fails on startup, set manager to None to prevent app from running
    graph_db_manager = None
