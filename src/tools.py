import json
from datetime import datetime
from .logger import logger

def get_current_datetime() -> str:
    """
    Returns the current date and time.

    Returns:
        str: A string containing the current date and time in ISO format.
    """
    logger.info("Executing tool 'get_current_datetime'")
    now = datetime.now()
    return now.isoformat()

# --- Tool Definitions for the LLM ---

# A mapping from the tool name (as the LLM knows it) to the actual Python function.
available_tools = {
    "get_current_datetime": get_current_datetime,
}

# The JSON schema that describes the tools to the LLM.
# This is how the LLM knows what tools are available, what they do, and what arguments they take.
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]
