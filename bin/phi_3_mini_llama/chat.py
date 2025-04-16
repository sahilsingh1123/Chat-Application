"""
This class contains the chat inference for
phi-3-mini API
"""

import os
from llama_cpp import Llama
from ..chat_interface import Chat
from dotenv import load_dotenv


load_dotenv()
MODEL_PATH = os.getenv("PHI_3_MINI_MODEL_PATH_LLAMA")
ASSISTANT_ROLE = os.getenv("ASSISTANT_ROLE")


# class Phi3MiniChat:
class Phi3MiniChatLlama(Chat):
    def __init__(self):
        # super().__init__()
        self._model = MODEL_PATH
        self.llm = self._get_client()

    def chat(self, msg):
        res = self.llm.create_chat_completion(
            messages=self._get_template(msg), max_tokens=100
        )
        return res.get("completion", "")

    def _get_client(self):
        return Llama(
            model_path=self._model,
            n_threads=4,  # Adjust based on your CPU configuration.
            n_ctx=4096,  # Maximum context length (model-specific; "4k" implies 4096 tokens).
            temperature=0.7,  # Adjust as needed.
        )

    def _get_template(self, msg):
        # return [
        #     {
        #         "role": "system",
        #         "content": ASSISTANT_ROLE
        #     },
        #     {
        #         "role": "user",
        #         "content": msg
        #     },
        # ]
        return f"""<s><|user|>
        {msg}<|end|>
        <|assistant|>"""


if __name__ == "__main__":
    chat = Phi3MiniChatLlama()
    response = chat.chat("Hello, how are you?")
    print(response)
