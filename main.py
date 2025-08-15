import asyncio
import json
from src.config import llm_client, LLM_MODEL_NAME
from src.conversation_processor import setup_conversation_database
from src.vector_store import conversation_vector_store, VectorCollectionManager
from src.graph_db_manager import graph_db_manager
from src.logger import logger
from src.context_manager import ContextManager
from src.tools import tools_schema, available_tools


async def main():
    """Main function to run the tool-calling agent.
    This version implements a full request-tool_call-response loop.
    """
    # --- 0. Setup ---
    if not graph_db_manager:
        logger.critical("Database connection failed. Exiting.")
        return

    await conversation_vector_store.setup()
    setup_conversation_database()

    context_manager = ContextManager()

    try:
        # --- 1. Start Interactive Loop ---
        logger.success("--- Tool-Calling Agent is Ready ---")
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
            await context_manager.add_user_message(query)

            # --- 3. Generate Response (with potential tool calls) ---
            logger.info("Generating response...")
            try:
                messages_for_llm = await context_manager.messages
                if not messages_for_llm:
                    logger.error("Could not generate messages for the LLM, skipping.")
                    continue

                # First API call to the LLM
                response = await llm_client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=messages_for_llm,
                    tools=tools_schema,  # Make tools available
                    tool_choice="auto",  # Let the model decide when to use tools
                    temperature=0.7,
                )
                response_message = response.choices[0].message

                # --- 4. Handle Tool Calls if any ---
                tool_calls = response_message.tool_calls
                if tool_calls:
                    logger.info(f"LLM requested to use {len(tool_calls)} tool(s).")
                    # The context manager needs to know about the initial tool_call request
                    await context_manager.add_assistant_message(response_message)

                    # Execute all tool calls
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_to_call = available_tools.get(function_name)

                        if not function_to_call:
                            logger.error(f"Unknown tool requested: {function_name}")
                            continue

                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            logger.info(
                                f"Calling tool: {function_name} with args: {function_args}"
                            )

                            # Call the actual tool function
                            function_response = function_to_call(**function_args)

                            # Add the tool's output to the context
                            await context_manager.add_tool_response(
                                tool_call.id, function_response
                            )
                            logger.success(
                                f"Tool {function_name} executed successfully."
                            )

                        except Exception as e:
                            logger.error(f"Error executing tool {function_name}: {e}")
                            await context_manager.add_tool_response(
                                tool_call.id, f"Error: {e}"
                            )

                    # --- 5. Second API call to get the final user-facing response ---
                    logger.info(
                        "Sending tool responses back to LLM for final answer..."
                    )
                    final_messages = await context_manager.messages
                    final_response = await llm_client.chat.completions.create(
                        model=LLM_MODEL_NAME,
                        messages=final_messages,
                        temperature=0.7,
                    )
                    final_answer = final_response.choices[0].message.content
                    logger.success(f"--- Final Answer ---{final_answer}")
                    # Run memory update in the background
                    asyncio.create_task(context_manager.add_assistant_message(final_answer))

                else:
                    # --- No tool calls, just a direct answer ---
                    final_answer = response_message.content
                    logger.success(f"--- Answer ---{final_answer}")
                    # Run memory update in the background
                    asyncio.create_task(context_manager.add_assistant_message(final_answer))

            except Exception as e:
                logger.error(f"An error occurred while generating the answer: {e}")

    finally:
        # --- 6. Cleanup ---
        await VectorCollectionManager.close_all()
        graph_db_manager.close()
        logger.info("--- System Shutting Down ---")


if __name__ == "__main__":
    asyncio.run(main())
