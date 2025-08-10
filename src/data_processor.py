import asyncio
import re
import numpy as np
import nltk
from langchain_core.documents import Document
from langchain_text_splitters import NLTKTextSplitter
from src.logger import logger
from src.config import embedding_client, EMBEDDING_MODEL_NAME

# --- NLTK Resource Download ---
# The NLTKTextSplitter requires resources like 'punkt'.
# This block ensures they are downloaded if they don't exist.
def download_nltk_resource_if_missing(resource_id, resource_path):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        logger.info(f"NLTK '{resource_id}' resource not found. Downloading...")
        nltk.download(resource_id, quiet=True)
        logger.info(f"NLTK '{resource_id}' resource downloaded successfully.")

download_nltk_resource_if_missing('punkt', 'tokenizers/punkt')
# -----------------------------

class CustomSemanticSplitter:
    """
    A custom text splitter that groups sentences into chunks based on semantic similarity.
    This implementation uses a LangChain sentence splitter and calculates cosine similarity
    between adjacent sentence embeddings to find topical breaks.
    """
    def __init__(self, client, model_name, breakpoint_threshold_percentile=95):
        self._client = client
        self._model_name = model_name
        self.breakpoint_threshold_percentile = breakpoint_threshold_percentile
        self._sentence_splitter = NLTKTextSplitter()

    def _calculate_cosine_similarity(self, v1, v2):
        """Calculates cosine similarity between two vectors."""
        v1 = np.array(v1)
        v2 = np.array(v2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    async def create_documents(self, texts: list[str]) -> list[Document]:
        """Splits a list of texts into semantic documents."""
        all_chunks = []
        for text in texts:
            chunks = await self.split_text(text)
            for chunk in chunks:
                all_chunks.append(Document(page_content=chunk))
        return all_chunks

    async def split_text(self, text: str) -> list[str]:
        """The core logic to split a single text into semantic chunks."""
        sentences = self._sentence_splitter.split_text(text)
        sentences = [s.strip() for s in sentences if s and s.strip()]
        
        if not sentences:
            logger.warning("No sentences found in text.")
            return []

        try:
            logger.info(f"Embedding {len(sentences)} sentences...")
            response = await self._client.embeddings.create(
                model=self._model_name,
                input=sentences
            )
            embeddings = [d.embedding for d in response.data]
            logger.info(f"Successfully embedded {len(sentences)} sentences.")
        except Exception as e:
            logger.error(f"Failed to embed sentences. Error: {e}")
            return [text]

        similarities = [
            self._calculate_cosine_similarity(embeddings[i], embeddings[i+1])
            for i in range(len(embeddings) - 1)
        ]
        
        if not similarities:
            return sentences

        breakpoint_threshold = np.percentile(similarities, self.breakpoint_threshold_percentile)
        logger.info(f"Calculated similarity breakpoint threshold: {breakpoint_threshold:.4f} (at {self.breakpoint_threshold_percentile}th percentile)")

        chunks = []
        current_chunk = []
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            if i < len(similarities) and similarities[i] < breakpoint_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return [chunk for chunk in chunks if chunk]

# --- Global Text Splitter Instance ---
_text_splitter = CustomSemanticSplitter(
    client=embedding_client,
    model_name=EMBEDDING_MODEL_NAME,
    breakpoint_threshold_percentile=95 
)
# -------------------------------------------------

async def load_and_split_text(file_path: str):
    """Loads text, then asynchronously splits it into semantic chunks in a non-blocking way."""
    try:
        def _read_file_sync():
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'[ \t]{2,}', ' ', text)
                return text.strip()
        text = await asyncio.to_thread(_read_file_sync)
    except FileNotFoundError:
        logger.error(f"File not found at {file_path}")
        return []

    if not text:
        logger.warning(f"No content in {file_path} after cleaning. Skipping.")
        return []

    # The splitting process is now fully async and can be awaited directly
    docs = await _text_splitter.create_documents([text])
    logger.info(f"Split text from {file_path} into {len(docs)} semantic chunks.")
    return docs

if __name__ == '__main__':
    async def main_test():
        print("--- Running CustomSemanticSplitter Test ---")
        
        test_text = (
            "The sun rises in the east, casting long shadows across the land. "
            "Birds begin to sing, welcoming the new day with their cheerful melodies. "
            "The sky is a brilliant blue, dotted with fluffy white clouds. "
            "Meanwhile, in the world of computing, a processor, or central processing unit (CPU), is the electronic circuitry that executes instructions comprising a computer program. "
            "The CPU performs basic arithmetic, logic, controlling, and input/output (I/O) operations specified by the instructions in the program. "
            "Moore's Law is the observation that the number of transistors in an integrated circuit (IC) doubles about every two years."
        )
        
        print("\n--- Input Text ---")
        print(test_text)
        
        print("\n--- Splitting Text ---")
        chunks = await _text_splitter.split_text(test_text)
        
        print(f"\n--- Found {len(chunks)} Semantic Chunks ---")
        for i, chunk in enumerate(chunks):
            print(f"--- Chunk {i+1} ---")
            print(chunk)
            print("-" * 20)
            
    asyncio.run(main_test())