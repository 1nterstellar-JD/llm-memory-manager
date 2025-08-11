import re
import asyncio
import numpy as np
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from src.logger import logger
from src.config import (
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    OPENAI_API_KEY,
    EMBEDDING_MODEL_URL,
)

# --- NLTK Setup ---
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except (ImportError, LookupError):
    logger.info("NLTK 'punkt' tokenizer not found. Downloading...")
    try:
        nltk.download("punkt", quiet=True)
        logger.info("'punkt' downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download NLTK 'punkt' tokenizer: {e}")
        nltk = None
# --------------------


# --- Embeddings and Similarity ---
embeddings_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=OPENAI_API_KEY,
    base_url=EMBEDDING_MODEL_URL,
)

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)

async def aget_embedding(text: str):
    """Gets the embedding for a given text asynchronously."""
    try:
        return await embeddings_model.aembed_query(text)
    except Exception as e:
        logger.error(f"Failed to get embedding for text: '{text}'. Error: {e}")
        return None
# ---------------------------------


async def _aadd_semantic_overlap(chunks: list[str], threshold: float, overlap_sentences: int) -> list[str]:
    """
    Post-processes chunks to add overlap based on semantic similarity (async).
    """
    if len(chunks) < 2 or nltk is None:
        return chunks

    final_chunks = []
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i+1]

        current_sents = nltk.sent_tokenize(current_chunk)
        next_sents = nltk.sent_tokenize(next_chunk)

        if not current_sents or not next_sents:
            final_chunks.append(current_chunk)
            continue

        last_sent_current = current_sents[-1]
        first_sent_next = next_sents[0]

        # Get embeddings concurrently
        v1, v2 = await asyncio.gather(
            aget_embedding(last_sent_current),
            aget_embedding(first_sent_next)
        )

        similarity = cosine_similarity(v1, v2)

        if similarity > threshold:
            overlap_content = " ".join(next_sents[:overlap_sentences])
            logger.debug(f"Found high similarity ({similarity:.2f}) between chunks {i} and {i+1}. Adding overlap.")
            current_chunk += " " + overlap_content
        
        final_chunks.append(current_chunk)

    final_chunks.append(chunks[-1])
    return final_chunks

async def load_and_split_text(file_path: str) -> list[Document]:
    """
    Loads text, splits it, and adds semantic overlap asynchronously.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
    except FileNotFoundError:
        logger.error(f"File not found at {file_path}")
        return []

    if not text:
        logger.warning(f"No content in {file_path} after cleaning. Skipping.")
        return []

    base_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    initial_chunks = base_splitter.split_text(text)
    logger.info(f"Split text from {file_path} into {len(initial_chunks)} initial chunks.")

    SIMILARITY_THRESHOLD = 0.7
    OVERLAP_SENTENCES = 1
    
    final_chunks = await _aadd_semantic_overlap(initial_chunks, SIMILARITY_THRESHOLD, OVERLAP_SENTENCES)
    
    docs = [Document(page_content=chunk) for chunk in final_chunks]
    logger.info(f"Processed into {len(docs)} final chunks with semantic overlap.")
    return docs


if __name__ == "__main__":

    async def main_test():
        print("--- Running Async Semantic Overlap Splitter Test ---")
        try:
            with open("data/test.txt", "r", encoding="utf-8") as f:
                test_text = f.read()
        except FileNotFoundError:
            print("Test file 'data/test.txt' not found.")
            return

        print("\n--- Splitting Text with Semantic Overlap ---")
        
        base_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        initial_chunks = base_splitter.split_text(test_text)
        print(f"\n--- Found {len(initial_chunks)} Initial Chunks ---")

        final_chunks = await _aadd_semantic_overlap(initial_chunks, threshold=0.7, overlap_sentences=1)
        
        print(f"\n--- Generated {len(final_chunks)} Final Chunks (with overlap) ---")
        for i, chunk in enumerate(final_chunks):
            print(f"--- Chunk {i+1} ---")
            print(chunk)
            print("-" * 20)

    asyncio.run(main_test())