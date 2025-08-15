import asyncio
import time
from src.config import (
    llm_client,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    CONVERSATION_TOKEN_THRESHOLD,
)
from src.conversation_processor import (
    process_conversation,
    count_tokens_in_messages,
    summarize_conversation,
    setup_conversation_database,
)
from src.retriever import retrieve_context, get_embedding_from_api
from src.vector_store import conversation_vector_store, VectorCollectionManager
from src.graph_db_manager import graph_db_manager
from src.logger import logger


async def main():
    """Main function to run the conversational memory application."""
    # --- 0. Setup ---
    if not graph_db_manager:
        logger.critical("Database connection failed. Exiting.")
        return

    await conversation_vector_store.setup()
    setup_conversation_database()

    try:
        # --- 1. Start Interactive Q&A Loop ---
        logger.success("--- Conversational Memory System is Ready ---")
        logger.info("Start chatting with the AI. Type 'exit' to quit.")

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

            # --- 2. Summarize conversation if it's too long ---
            token_count = count_tokens_in_messages(conversation_history)
            if token_count > CONVERSATION_TOKEN_THRESHOLD:
                summary = await summarize_conversation(conversation_history)
                if "Error:" not in summary:
                    conversation_history = [
                        {
                            "role": "system",
                            "content": f"This is a summary of the previous conversation: {summary}",
                        }
                    ]
                    logger.info(
                        f"Conversation summarized. New token count: {count_tokens_in_messages(conversation_history)}"
                    )
                else:
                    logger.error(
                        "Could not replace history with summary due to an error."
                    )

            conversation_history.append({"role": "user", "content": query})

            # --- 3. Retrieve Context from past conversations ---
            logger.info("Retrieving context from memory...")
            context = await retrieve_context(query)

            if "Error:" in context:
                logger.error(context)
                context = "" # Don't stop, just proceed without context

            if context:
                logger.info(
                    f"--- Retrieved Context ---\n{context}\n--------------------------"
                )

            # --- 4. Generate Answer with LLM ---
            logger.info("Generating answer...")
            try:
                # Construct the prompt with or without context
                prompt_messages = []
                if context:
                    prompt_messages.append({"role": "system", "content": f"Please use the following context to answer the user's question.\n\nContext:\n{context}"})

                # Add recent history to the prompt for short-term memory
                # For this simplified version, we just use the current query.
                # A more advanced version might include the last few turns.
                prompt_messages.append({"role": "user", "content": query})

                response = await llm_client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=prompt_messages,
                    temperature=0.7,
                )
                answer = response.choices[0].message.content
                logger.success(f"--- Answer ---\n{answer}")

                conversation_history.append({"role": "assistant", "content": answer})

                # --- 5. Post-processing: Update long-term memory ---
                # Process the last turn (user query + AI answer) to update the graph
                await process_conversation(conversation_history[-2:])

                # Store the turn in the vector store for semantic search
                try:
                    turn_text = f"User: {query}\nAssistant: {answer}"
                    turn_embedding = await get_embedding_from_api(
                        turn_text, EMBEDDING_MODEL_NAME
                    )
                    if turn_embedding:
                        turn_id = f"{int(time.time())}-{len(conversation_history)}"
                        await conversation_vector_store.insert_batch(
                            [
                                {
                                    "id": turn_id,
                                    "embedding": turn_embedding,
                                    "text": turn_text,
                                }
                            ]
                        )
                        logger.info(
                            f"Stored conversation turn '{turn_id}' in vector store."
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to store conversation turn in vector store: {e}"
                    )

            except Exception as e:
                logger.error(f"An error occurred while generating the answer: {e}")
    finally:
        # --- 6. Cleanup ---
        await VectorCollectionManager.close_all()
        graph_db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
