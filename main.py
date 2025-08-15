import os
import asyncio
import time
from src.config import (
    llm_client,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    CONVERSATION_TOKEN_THRESHOLD
)
from src.data_processor import load_and_split_text
from src.graph_builder import build_graph_from_chunks, setup_database_constraints
from src.conversation_processor import (
    process_conversation,
    count_tokens_in_messages,
    summarize_conversation,
)
from src.retriever import (
    retrieve_context,
    sync_embeddings_with_vector_store,
    get_embedding_from_api
)
from src.vector_store import (
    doc_vector_store,
    conversation_vector_store,
    VectorCollectionManager
)
from src.graph_db_manager import graph_db_manager
from src.logger import logger

async def process_new_data(source_file: str):
    """Loads text, splits it, and builds the knowledge graph in Neo4j."""
    logger.info(f"Processing source file: {source_file}")
    chunks = await load_and_split_text(source_file)
    if not chunks:
        logger.error("No text chunks were created. Aborting data processing.")
        return False

    await build_graph_from_chunks(chunks)
    logger.info("Graph construction complete. Syncing embeddings with vector store...")
    await sync_embeddings_with_vector_store()
    logger.success("Data processing and graph building finished successfully.")
    return True

async def main():
    """Main function to run the Graph RAG application with Neo4j."""
    # --- 0. Setup ---
    await doc_vector_store.setup()
    await conversation_vector_store.setup()
    setup_database_constraints()

    try:
        # --- 1. Check if Graph is Empty and Build if Necessary ---
        result = graph_db_manager.execute_query("MATCH (n) RETURN count(n) as count")
        node_count = result.single()["count"] if result else 0

        if node_count == 0:
            logger.info("Knowledge graph is empty. Building from source data...")
            source_file = "data/test.txt"
            if not os.path.exists(source_file):
                logger.error(f"Source data file not found at {source_file}")
                return
            await process_new_data(source_file)
        else:
            logger.info(f"Knowledge graph already contains {node_count} nodes.")
            # Ensure vector store is in sync with the graph
            await sync_embeddings_with_vector_store()


        # --- 2. Start Interactive Q&A Loop ---
        logger.success("--- Knowledge Graph RAG is Ready ---")
        logger.info("Ask a question about the content. Type 'exit' to quit.")

        conversation_history = []

        while True:
            try:
                query = await asyncio.to_thread(input, "\nYour question: ")
            except KeyboardInterrupt:
                break

            if query.lower() == "exit":
                break
            if not query.strip():
                continue

            # --- 3. Summarize conversation if it's too long ---
            token_count = count_tokens_in_messages(conversation_history)
            if token_count > CONVERSATION_TOKEN_THRESHOLD:
                summary = await summarize_conversation(conversation_history)
                if "Error:" not in summary:
                    conversation_history = [
                        {"role": "system", "content": f"This is a summary of the previous conversation: {summary}"}
                    ]
                    logger.info(f"Conversation summarized. New token count: {count_tokens_in_messages(conversation_history)}")
                else:
                    logger.error("Could not replace history with summary due to an error.")

            conversation_history.append({"role": "user", "content": query})

            # --- 4. Retrieve Context from the Graph ---
            logger.info("Retrieving context from the graph...")
            context = await retrieve_context(query)

            if "Error:" in context:
                logger.error(context)
                continue

            logger.info(f"--- Retrieved Context ---\n{context}\n--------------------------")

            # --- 4. Generate Answer with LLM ---
            logger.info("Generating answer...")
            try:
                prompt = f"""Based on the following context, please answer the user's question.
                If the context does not contain the answer, say that you cannot find the information in the provided text.

                Context:
                {context}

                Question: {query}

                Answer:"""

                response = await llm_client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions based on provided context.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                )
                answer = response.choices[0].message.content
                logger.success(f"--- Answer ---\n{answer}")

                conversation_history.append({"role": "assistant", "content": answer})

                # --- 4a. Process conversation to update graph ---
                await process_conversation(conversation_history[-2:])

                # --- 4b. Store conversation turn in vector store for semantic search ---
                try:
                    turn_text = f"User: {query}\nAssistant: {answer}"
                    turn_embedding = await get_embedding_from_api(turn_text, EMBEDDING_MODEL_NAME)
                    if turn_embedding:
                        turn_id = f"{int(time.time())}-{len(conversation_history)}"
                        await conversation_vector_store.insert_batch(
                            [{"id": turn_id, "embedding": turn_embedding, "text": turn_text}]
                        )
                        logger.info(f"Stored conversation turn '{turn_id}' in vector store.")
                except Exception as e:
                    logger.error(f"Failed to store conversation turn in vector store: {e}")

            except Exception as e:
                logger.error(f"An error occurred while generating the answer: {e}")
    finally:
        # --- 5. Cleanup ---
        await VectorCollectionManager.close_all()
        graph_db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
