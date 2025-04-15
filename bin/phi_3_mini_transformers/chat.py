"""
This class contains the chat interface for
phi-3-mini API with hugging face transformers
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from bin.chat_interface import Chat
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_PATH = os.getenv('PHI_3_MINI_MODEL_PATH_HF')
ASSISTANT_ROLE = os.getenv('ASSISTANT_ROLE')
DEVICE = os.getenv('DEVICE')


class Phi3MiniChatHF(Chat):
    def __init__(self):
        super().__init__()
        self._model = MODEL_PATH
        self.llm = self._get_client()
        self.tokenizers = AutoTokenizer.from_pretrained(self._model)

    def _get_client(self):
        return AutoModelForCausalLM.from_pretrained(
            self._model,
            device_map=DEVICE,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )

    def chat(self, msg):
        tokens = self._get_tokens(msg)
        output_tokens = self.llm.generate(**tokens, max_new_tokens=200, do_sample=False)
        return self._decode_texts(output_tokens)

    def _get_tokens(self, msg):
        # call get_template method
        prompt = self._get_template(msg)
        device = torch.device("mps")
        return self.tokenizers(msg, return_tensors="pt").to(device)

    def _decode_texts(self, output_tokens):
        return self.tokenizers.decode(output_tokens[0], skip_special_tokens=True)

    def _get_template(self, msg):
        return f"""<s><|user|>
            {msg}<|end|>
            <|assistant|>"""

if __name__ == "__main__":
    chat = Phi3MiniChatHF()
    print(chat.chat("Hello"))