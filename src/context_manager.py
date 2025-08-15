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
            if "Error:" not in summary:
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

    async def add_assistant_message(self, answer: str):
        """
        Adds an assistant's message to the history and triggers long-term memory updates.
        """
        # 1. Add assistant answer to history
        self.history.append({"role": "assistant", "content": answer})
        logger.info("Added assistant message to history.")

        # 2. Get the last turn (user query + AI answer)
        last_turn = self.history[-2:]
        if not (len(last_turn) == 2 and last_turn[0]["role"] == "user"):
            logger.warning("Could not process last turn for memory update.")
            return

        # 3. Update long-term memory (Graph and Vector Store)
        await process_conversation(last_turn)

        try:
            query = last_turn[0]["content"]
            turn_text = f"User: {query}\nAssistant: {answer}"
            turn_embedding = await get_embedding_from_api(
                turn_text, EMBEDDING_MODEL_NAME
            )
            if turn_embedding:
                turn_id = f"{int(time.time())}-{len(self.history)}"
                await conversation_vector_store.insert_batch(
                    [{"id": turn_id, "embedding": turn_embedding, "text": turn_text}]
                )
                logger.info(f"Stored conversation turn '{turn_id}' in vector store.")
        except Exception as e:
            logger.error(f"Failed to store conversation turn in vector store: {e}")

    @property
    async def messages(self) -> List[Dict[str, Any]]:
        """
        Constructs the list of messages to be sent to the LLM, including retrieved context.
        This is the property that will be passed to the OpenAI client.
        """
        if not self.history or self.history[-1]["role"] != "user":
            logger.warning(
                "Cannot generate messages: history is empty or last message is not from user."
            )
            return self.history

        query = self.history[-1]["content"]

        # 1. Retrieve context from memory
        logger.info(f"Retrieving context for query: '{query}'")
        context = await retrieve_context(query)
        if "Error:" in context:
            logger.error(context)
            context = ""  # Proceed without context on error

        if context:
            logger.info(
                f"--- Retrieved Context ---\n{context}\n--------------------------"
            )

        # 2. Construct the prompt messages
        prompt_messages = []
        if context:
            prompt_messages.append(
                {
                    "role": "system",
                    "content": f"Please use the following context to answer the user's question.\n\nContext:\n{context}",
                }
            )
        
        # Include recent history for short-term memory.
        # For now, we add the full history. A more advanced implementation
        # could be more selective.
        prompt_messages.extend(self.history)

        # A common pattern is to have one system message. Let's ensure that.
        # We will merge our context system message with any existing system message (e.g., from summarization)
        
        final_messages = []
        system_contents = []
        
        for msg in prompt_messages:
            if msg["role"] == "system":
                system_contents.append(msg["content"])
            else:
                final_messages.append(msg)

        if system_contents:
            final_system_message = "\n\n---\n\n".join(system_contents)
            final_messages.insert(0, {"role": "system", "content": final_system_message})

        return final_messages
