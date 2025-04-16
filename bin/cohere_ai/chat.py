"""
This class contains the chat interface for
cohere API
Chat models available:
    - command-r7b-12-2024
    - command-a-03-2025
"""

import asyncio
import functools
import os
import cohere
from dotenv import load_dotenv
from ..chat_interface import Chat

load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")
MODEL_NAME = os.getenv("COHERE_MODEL_NAME")
ASSISTANT_ROLE = os.getenv("ASSISTANT_ROLE")


class CohereChat(Chat):
    def __init__(self):
        super().__init__()
        self.co = self._get_client()
        self._model = MODEL_NAME

    def chat(self, msg):
        res = self.co.chat(model=self._model, messages=self._get_template(msg))

        return res.message.content[0].text

    def sync_stream_chat(self, msg):
        # This uses the synchronous generator provided by the cohere API
        response = self.co.chat_stream(
            model=self._model, messages=self._get_template(msg)
        )
        # for event in response:
        #     # Check the event and accumulate content
        #     if event and event.type == 'content-delta':
        #         yield event.delta.message.content.text
        return response

    async def stream_chat(self, msg):
        loop = asyncio.get_running_loop()
        # Run the synchronous chat stream in a thread so as not to block the event loop
        result = await loop.run_in_executor(
            None, functools.partial(self.sync_stream_chat, msg)
        )
        return result

    def _get_client(self):
        return cohere.ClientV2(API_KEY)

    def _get_template(self, msg):
        return [
            {"role": "user", "content": msg},
            {"role": "system", "content": ASSISTANT_ROLE},
        ]
