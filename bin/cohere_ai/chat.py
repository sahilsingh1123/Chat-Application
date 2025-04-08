"""
This class contains the chat interface for
cohere API
Chat models available:
    - command-r7b-12-2024
    - command-a-03-2025
"""
import os
import cohere
from dotenv import load_dotenv
from ..chat_interface import Chat

load_dotenv()
API_KEY = os.getenv('COHERE_API_KEY')
MODEL_NAME = os.getenv('COHERE_MODEL_NAME')
ASSISTANT_ROLE = os.getenv('ASSISTANT_ROLE')


class CohereChat(Chat):
    def __init__(self):
        super().__init__()
        self.co = self._get_client()
        self._model = MODEL_NAME

    def chat(self, msg):
        res = self.co.chat(
            model=self._model,
            messages=self._get_template(msg)
        )

        return res.message.content[0].text

    def _get_client(self):
        return cohere.ClientV2(API_KEY)

    def _get_template(self, msg):
        return [
            {
                "role": "user",
                "content": msg
            },
            {
                "role": "assistant",
                "content": ASSISTANT_ROLE
            }
        ]
