"""
This class contains the chat interface for
tiny-llama API
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from bin.chat_interface import Chat
from bin.chat_providers import ChatFactory
from bin.chat_history_service import chat_history
from bin.constant import TINY_LLAMA

from dotenv import load_dotenv
import os
import re

load_dotenv()
MODEL_PATH = os.getenv("TINY_LLAMA_MODEL_NAME")
ASSISTANT_ROLE = os.getenv("ASSISTANT_ROLE")
DEVICE = os.getenv("DEVICE")


@ChatFactory.register(TINY_LLAMA)
class TinyLlamaChat(Chat):
    def __init__(self):
        super().__init__()
        self._model = MODEL_PATH
        self._device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._llm = self._get_client()
        self._tokenizers = self._get_tokenizers()

    def _get_client(self):
        return AutoModelForCausalLM.from_pretrained(
            self._model, device_map=DEVICE
        )

    def _get_tokenizers(self):
        return AutoTokenizer.from_pretrained(
            self._model,
            device_map=DEVICE,
        )

    def _get_template(self, msg):
        # prompt = ""
        # for message in chat_history.get_history():
        #     prompt += f"<|{message['role']}|>\n{message['content']}\n"
        # prompt += "<|assistant|>\n"
        prompt = chat_history.get_formatted_history()
        prompt += f"<|user|>\n{msg}<s>\n"
        prompt += "<|assistant|>\n"
        print("Prompt:", prompt)
        return prompt

    def chat(self, msg):
        tokens = self._tokenize_text(msg)
        output_ids = self._llm.generate(
            tokens,
            max_new_tokens=500,  # Maximum number of tokens the model can add
            do_sample=False,  # Enable sampling for a more diverse output
            temperature=0.7,  # Controls randomness (0.0 deterministic, higher more random)
            top_k=50,  # Consider only the top k probabilities
            top_p=0.95,  # Nucleus sampling: consider tokens until the cumulative probability reaches this value
            eos_token_id=self._tokenizers.eos_token_id,  # End-of-sequence token to stop generation
        )
        return self._decode_response(output_ids)

    def _tokenize_text(self, msg):
        return self._tokenizers(
            self._get_template(msg), return_tensors="pt"
        ).input_ids.to(self._device)

    @staticmethod
    def _extract_assistant_response(text):
        # Regular expression to match content after <|assistant|> up to the next role tag or end of string
        match = re.search(
            r"<\|assistant\|>\s*(.*?)(?=\n<\|[a-zA-Z]+\|>|$)", text, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return None

    def _decode_response(self, output_ids):
        decoded_text = self._tokenizers.decode(
            output_ids[0], skip_special_tokens=True
        )
        return self._extract_assistant_response(decoded_text)


if __name__ == "__main__":
    chat = TinyLlamaChat()
    print(chat.chat("Tell me about flask framework in python"))
