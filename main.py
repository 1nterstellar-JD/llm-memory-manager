import asyncio
from src.config import llm_client, LLM_MODEL_NAME
from src.conversation_processor import setup_conversation_database
from src.vector_store import conversation_vector_store, VectorCollectionManager
from src.graph_db_manager import graph_db_manager
from src.logger import logger
from src.context_manager import ContextManager


async def main():
    """Main function to run the conversational memory application."""
    # --- 0. Setup ---
    if not graph_db_manager:
        logger.critical("Database connection failed. Exiting.")
        return

    await conversation_vector_store.setup()
    setup_conversation_database()

    # Initialize the context manager that will handle the conversation logic
    context_manager = ContextManager()

    try:
        # --- 1. Start Interactive Q&A Loop ---
        logger.success("--- Conversational Memory System is Ready ---")
        logger.info("Start chatting with the AI. Type 'exit' to quit.")

        while True:
            try:
                query = await asyncio.to_thread(input, "\nYour question: ")
            except KeyboardInterrupt:
                break

            if query.lower() == "exit":
                break
            if not query.strip():
                continue

            # --- 2. Add user message to context ---
            # The manager will handle history, summarization, etc.
            await context_manager.add_user_message(query)

            # --- 3. Generate Answer with LLM ---
            logger.info("Generating answer...")
            try:
                # The context_manager.messages property dynamically builds the prompt
                # with the latest context and history.
                messages_for_llm = await context_manager.messages

                if not messages_for_llm:
                    logger.error("Could not generate messages for the LLM, skipping.")
                    continue

                response = await llm_client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=messages_for_llm,
                    temperature=0.7,
                )
                answer = response.choices[0].message.content
                logger.success(f"--- Answer ---\n{answer}")

                # --- 4. Add assistant answer and update memories ---
                # The manager will handle updating the graph and vector store.
                await context_manager.add_assistant_message(answer)
                logger.success(await context_manager.messages)

            except Exception as e:
                logger.error(f"An error occurred while generating the answer: {e}")

    finally:
        # --- 5. Cleanup ---
        await VectorCollectionManager.close_all()
        graph_db_manager.close()
        logger.info("--- System Shutting Down ---")


if __name__ == "__main__":
    asyncio.run(main())
