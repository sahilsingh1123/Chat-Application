"""
This class contains the chat interface for
tiny-llama API
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from llama_cpp import Llama

from bin.chat_interface import Chat
from bin.chat_providers import ChatFactory
from bin.chat_history_service import chat_history
from bin.constant import DEEPSEEK

from torch.quantization import quantize_dynamic
from dotenv import load_dotenv
import os
import re

load_dotenv()
MODEL_PATH = os.getenv("DEEPSEEK_MODEL_NAME")
MODEL_PATH_GGUF = os.getenv("DEEPSEEK_MODEL_NAME_GGUF")
ASSISTANT_ROLE = os.getenv("ASSISTANT_ROLE")
DEVICE = os.getenv("DEVICE")


@ChatFactory.register(DEEPSEEK)
class DeepSeekChat(Chat):
    def __init__(self):
        super().__init__()
        self._model = MODEL_PATH
        self._use_metal = True  # Use Metal GPU
        self._n_ctx = 2048  # Context window size
        self._n_threads = 4  # Number of CPU threads
        self._use_gguf = True
        self._device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._llm = self._get_client()
        self._tokenizers = self._get_tokenizers()

    def _get_client(self):
        if self._use_gguf:
            return self._get_client_llama()
        return self._get_client_basic()

    def _get_client_llama(self):
        return Llama(
            model_path=MODEL_PATH_GGUF,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            use_metal=self._use_metal,
        )

    def _get_client_basic(self):
        return AutoModelForCausalLM.from_pretrained(
            self._model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=DEVICE,
        )

    def _get_client_bitsandbytes(self):
        # configure for bitsnbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return AutoModelForCausalLM.from_pretrained(
            self._model,
            device_map=DEVICE,
            quantization_config=bnb_config,
            use_flash_attention=True,
            torch_dtype=torch.bfloat16,
        )

    def _get_client_pytorch(self):
        model = AutoModelForCausalLM.from_pretrained(
            self._model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=DEVICE,
        )
        return quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

    def _get_tokenizers(self):
        return AutoTokenizer.from_pretrained(
            self._model,
            device_map=DEVICE,
        )

    def _get_template(self, msg):
        pass

    def _tokenize_text(self, msg):
        return self._tokenizers(msg, return_tensors="pt").to(self._device)

    def _decode_response(self, output_ids):
        decoded_text = self._tokenizers.decode(
            output_ids[0], skip_special_tokens=True
        )
        return decoded_text

    def chat(self, msg):
        if self._use_gguf:
            return self.chat_gguf_llama(msg)
        output_ids = self._llm.generate(
            self._tokenize_text(msg),
            max_new_tokens=256,  # Maximum number of tokens the model can add
            do_sample=True,  # Enable sampling for a more diverse output
            temperature=0.7,  # Controls randomness (0.0 deterministic, higher more random)
            top_k=50,  # Consider only the top k probabilities
            top_p=0.95,  # Nucleus sampling: consider tokens until the cumulative probability reaches this value
            eos_token_id=self._tokenizers.eos_token_id,  # End-of-sequence token to stop generation
        )

        return self._decode_response(output_ids)

    def chat_gguf_llama(self, msg):
        try:
            response = self._llm(
                prompt=msg,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>"],
                stream=False,
            )
            return response.get("choices", [{}])[0].get("text", "").strip()
        except Exception as e:
            # Introspect and offer guidance
            raise RuntimeError("Inference failed: " + str(e))
