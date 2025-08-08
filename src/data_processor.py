from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import logger
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_text(file_path: str):
    """Loads text from a file and splits it into chunks based on config."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        logger.error(f"File not found at {file_path}")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    logger.info(f"Split text from {file_path} into {len(chunks)} chunks.")
    return chunks
