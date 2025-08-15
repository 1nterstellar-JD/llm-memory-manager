import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# To run these tests, ensure the project root is in your PYTHONPATH.
# For example: export PYTHONPATH=$PYTHONPATH:/path/to/your/project
# This allows the test runner to find the 'src' module.

from src.conversation_processor import extract_relationships_from_conversation

class TestConversationProcessor(unittest.TestCase):

    def test_extract_relationships_llm_call(self):
        """
        Tests the extract_relationships_from_conversation function's ability
        to correctly parse a mocked LLM response.
        """
        # Mock the response from the llm_client
        mock_response_content = {
            "relationships": [
                ["jules", "works at", "acme corp"],
                ["acme corp", "is a", "company"],
                ["acme corp", "located in", "new york"]
            ]
        }

        # The actual response object is nested
        mock_choice = MagicMock()
        mock_choice.message.content = str(mock_response_content).replace("'", '"') # Ensure it's a valid JSON string

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        # Since the function is async, we need an AsyncMock
        mock_llm_client = AsyncMock()
        mock_llm_client.chat.completions.create.return_value = mock_completion

        # Patch the llm_client used in the conversation_processor module
        with patch('src.conversation_processor.llm_client', mock_llm_client):
            # Run the async function
            result = asyncio.run(extract_relationships_from_conversation("Test text"))

            # Define expected result (normalized: lowercase and stripped)
            expected = [
                ["jules", "works at", "acme corp"],
                ["acme corp", "is a", "company"],
                ["acme corp", "located in", "new york"]
            ]

            # Assertions
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 3)
            self.assertListEqual(sorted(result), sorted(expected))

            # Check that the mocked client was called
            mock_llm_client.chat.completions.create.assert_called_once()

if __name__ == '__main__':
    unittest.main()
