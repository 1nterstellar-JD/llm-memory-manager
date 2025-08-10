import os
import asyncio
from src.config import llm_client, LLM_MODEL_NAME, GRAPH_OUTPUT_PATH
from src.data_processor import load_and_split_text
from src.graph_builder import build_graph_from_chunks
from src.graph_manager import save_graph, load_graph
from src.retriever import retrieve_context
from src.vector_store import vector_store
from src.logger import logger

async def main():
    """Main function to run the Graph RAG application."""
    # --- 0. Setup Vector Store ---
    await vector_store.setup()

    try:
        # --- 1. Load or Build the Knowledge Graph ---
        knowledge_graph = load_graph(GRAPH_OUTPUT_PATH)

        if knowledge_graph is None:
            logger.info("Building new knowledge graph from source data...")
            source_file = "data/test.txt"
            if not os.path.exists(source_file):
                logger.error(f"Source data file not found at {source_file}")
                return

            chunks = await load_and_split_text(source_file)
            if not chunks:
                logger.error("No text chunks were created. Exiting.")
                return

            knowledge_graph = await build_graph_from_chunks(chunks)
            save_graph(knowledge_graph, GRAPH_OUTPUT_PATH)

        # --- 2. Start Interactive Q&A Loop ---
        logger.success("--- Knowledge Graph RAG is Ready ---")
        logger.info("Ask a question about the content. Type 'exit' to quit.")

        while True:
            try:
                query = await asyncio.to_thread(input, "\nYour question: ")
            except KeyboardInterrupt:
                break

            if query.lower() == "exit":
                break
            if not query.strip():
                continue

            # --- 3. Retrieve Context from the Graph ---
            logger.info("Retrieving context from the graph...")
            context = await retrieve_context(query, knowledge_graph)

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

            except Exception as e:
                logger.error(f"An error occurred while generating the answer: {e}")
    finally:
        # --- 5. Cleanup ---
        await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())

