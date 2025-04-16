"""
This class contains the chat interface for
openai API
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
from ..chat_interface import Chat
from bin.constant import OPENAI
from bin.chat_providers import ChatFactory

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
ASSISTANT_ROLE = os.getenv("ASSISTANT_ROLE")


@ChatFactory.register(OPENAI)
class OpenAIChat(Chat):
    def __init__(self):
        super().__init__()
        self._openai = self._get_client()
        self._model = MODEL_NAME

    def chat(self, msg):
        res = self._openai.chat.completions.create(
            messages=self._get_template(msg),
            model=self._model,
        )

        return res.choices[0].message.content

    def _get_template(self, msg):
        return [
            {"role": "system", "content": ASSISTANT_ROLE},
            {"role": "user", "content": msg},
        ]

    def _get_client(self):
        return OpenAI(
            api_key=API_KEY,
        )
