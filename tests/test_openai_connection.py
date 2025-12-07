import unittest
import os
from dotenv import load_dotenv
from openai import OpenAI

class TestOpenAIConnection(unittest.TestCase):
    def test_openai_key_loaded(self):
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        self.assertIsNotNone(key, "OPENAI_API_KEY is not loaded")
        self.assertTrue(key.startswith("sk-"), "API key format is invalid")

    def test_openai_chat(self):
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        text = response.choices[0].message["content"]
        self.assertTrue(len(text) > 0, "Empty response from OpenAI")

if __name__ == "__main__":
    unittest.main()
