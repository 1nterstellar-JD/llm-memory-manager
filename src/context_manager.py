import asyncio
import time
from typing import List, Dict, Any

from .config import (
    CONVERSATION_TOKEN_THRESHOLD,
    EMBEDDING_MODEL_NAME,
)
from .conversation_processor import (
    process_conversation,
    count_tokens_in_messages,
    summarize_conversation,
)
from .retriever import retrieve_context, get_embedding_from_api
from .vector_store import conversation_vector_store
from .logger import logger


class ContextManager:
    """
    A class to manage the conversation context, including history,
    retrieval, and memory updates, for a conversational AI.
    """

    def __init__(self):
        """Initializes the ContextManager."""
        self.history: List[Dict[str, Any]] = []
        logger.info("ContextManager initialized.")

    async def add_user_message(self, query: str):
        """
        Adds a user message to the history and prepares for the next LLM call.
        This includes summarizing the conversation if it's too long.
        """
        # 1. Summarize conversation if it's too long
        token_count = count_tokens_in_messages(self.history)
        if token_count > CONVERSATION_TOKEN_THRESHOLD:
            summary = await summarize_conversation(self.history)
            # We filter out tool-related messages from the summary to keep it clean.
            history_to_summarize = [
                m
                for m in self.history
                if m.get("role") in ["user", "assistant"] and m.get("content")
            ]
            summary = await summarize_conversation(history_to_summarize)

            if "Error:" not in summary:
                # Replace the history with a summary message
                self.history = [
                    {
                        "role": "system",
                        "content": f"This is a summary of the previous conversation: {summary}",
                    }
                ]
                logger.info(
                    f"Conversation summarized. New token count: {count_tokens_in_messages(self.history)}"
                )
            else:
                logger.error("Could not replace history with summary due to an error.")

        # 2. Add the new user query to the history
        self.history.append({"role": "user", "content": query})
        logger.info(f"Added user message to history: '{query}'")

    async def add_assistant_message(self, message: Any):
        """
        Adds an assistant's message to the history. If it's the final text answer,
        it also triggers long-term memory updates.

        Args:
            message (Any): Can be a string (for final answers) or a message object
                         (e.g., containing tool_calls).
        """
        # 1. Convert message to standard dict format and add to history
        if isinstance(message, str):
            assistant_message = {"role": "assistant", "content": message}
        else:  # It's likely a ChatCompletionMessage object
            assistant_message = message.model_dump()

        self.history.append(assistant_message)
        logger.info("Added assistant message to history.")

        # 2. If this is a final answer (not a tool call), update long-term memory.
        if not assistant_message.get("tool_calls"):
            # Get the last full turn for memory processing.
            # This is more complex now with tool calls, so we find the last user message
            # and the final assistant answer.
            last_user_msg_index = -1
            for i in range(len(self.history) - 2, -1, -1):
                if self.history[i]["role"] == "user":
                    last_user_msg_index = i
                    break

            if last_user_msg_index != -1:
                # The turn includes the user message, any tool interactions, and the final answer.
                last_turn = self.history[last_user_msg_index:]
                await self.update_long_term_memory(last_turn)
            else:
                logger.warning(
                    "Could not find last user message to form a complete turn."
                )

    async def add_tool_response(self, tool_call_id: str, content: str):
        """
        Adds a tool's response message to the history.
        """
        message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        self.history.append(message)
        logger.info(f"Added tool response for call ID {tool_call_id}.")

    async def update_long_term_memory(self, turn_messages: List[Dict[str, Any]]):
        """
        Processes a list of messages representing a full turn and updates the
        Graph and Vector Stores.
        """
        logger.info(
            f"Updating long-term memory for a turn with {len(turn_messages)} messages."
        )

        # 1. Update Knowledge Graph
        # We combine the text content to extract relationships from the full context of the turn.
        turn_text_content = " ".join(
            [str(msg.get("content")) for msg in turn_messages if msg.get("content")]
        )
        await process_conversation([{"role": "user", "content": turn_text_content}])

        # 2. Update Vector Store
        try:
            # We create a single embedding for the entire turn to capture its full context.
            query = next(
                (msg["content"] for msg in turn_messages if msg["role"] == "user"), ""
            )
            answer = next(
                (
                    msg["content"]
                    for msg in turn_messages
                    if msg["role"] == "assistant" and msg.get("content")
                ),
                "",
            )

            if not query or not answer:
                logger.warning(
                    "Could not find user query or assistant answer in turn, skipping vector store update."
                )
                return

            turn_text_for_embedding = f"User: {query}\nAssistant: {answer}"
            turn_embedding = await get_embedding_from_api(
                turn_text_for_embedding, EMBEDDING_MODEL_NAME
            )
            if turn_embedding:
                turn_id = f"{int(time.time())}-{len(self.history)}"
                await conversation_vector_store.insert_batch(
                    [
                        {
                            "id": turn_id,
                            "embedding": turn_embedding,
                            "text": turn_text_for_embedding,
                        }
                    ]
                )
                logger.info(f"Stored conversation turn '{turn_id}' in vector store.")
        except Exception as e:
            logger.error(f"Failed to store conversation turn in vector store: {e}")

    @property
    async def messages(self) -> List[Dict[str, Any]]:
        """
        Constructs the list of messages for a tool-calling LLM, including retrieved context.
        """
        if not self.history:
            return []

        # The last message should be from the user to trigger a response.
        if self.history[-1]["role"] != "user":
            logger.warning("Last message is not from user, will not generate response.")
            return self.history

        query = self.history[-1]["content"]

        # 1. Retrieve context from long-term memory (Graph + Vector)
        logger.info(f"Retrieving context for query: '{query}'")
        retrieved_context = await retrieve_context(query)
        if "Error:" in retrieved_context:
            logger.error(retrieved_context)
            retrieved_context = ""  # Proceed without context on error

        if retrieved_context:
            logger.info(
                f"--- Retrieved Context ---\n{retrieved_context}\n--------------------------"
            )

        # 2. Construct the system prompt for the tool-calling agent
        # This prompt is now more explicit about the agent's capabilities and available information.
        system_prompt_parts = [
            "You are a helpful assistant that has access to a set of tools.",
            "You can use these tools to answer questions and perform tasks.",
        ]

        # Add any summary of the conversation history as a system-level note.
        # This is more effective than having it as a separate message.
        summary_message = next(
            (
                msg
                for msg in self.history
                if msg["role"] == "system" and "summary" in msg["content"]
            ),
            None,
        )
        if summary_message:
            system_prompt_parts.append(
                f"\n--- Summary of Previous Conversation ---\n{summary_message['content']}"
            )

        # Add the retrieved context (from KG and Vector DB) as background information.
        if retrieved_context:
            system_prompt_parts.append(
                f"\n--- Background Information ---\n"
                f"Here is some information retrieved from long-term memory that might be relevant to the user's query. "
                f"Use this to inform your decisions and tool usage.\n\n{retrieved_context}"
            )

        final_system_prompt = "\n".join(system_prompt_parts)

        # 3. Construct the final list of messages
        # We will have one system message, followed by the conversational history.
        final_messages = [{"role": "system", "content": final_system_prompt}]

        # Add the rest of the history, filtering out the old summary message
        # as it's now part of the main system prompt.
        for msg in self.history:
            if not (msg["role"] == "system" and "summary" in msg["content"]):
                final_messages.append(msg)

        return final_messages
