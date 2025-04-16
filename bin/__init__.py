"""
Import all chat providers to register them in ChatFactory
"""

from .cohere_ai.chat import CohereChat
from .openai.chat import OpenAIChat
from .phi_3_mini_llama.chat import Phi3MiniChatLlama
from .phi_3_mini_transformers.chat import Phi3MiniChatHF
from .tiny_llama.chat import TinyLlamaChat
