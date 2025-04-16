# from .cohere_ai.chat import CohereChat
# from .openai.chat import OpenAIChat
# from .phi_3_mini_llama.chat import Phi3MiniChatLlama
# from .phi_3_mini_transformers.chat import Phi3MiniChatHF
# from .tiny_llama.chat import TinyLlamaChat


# class ChatFactoryOld:
#     @staticmethod
#     def get_chat(chat_type):
#         if chat_type == "cohere":
#             return CohereChat()
#         elif chat_type == "openai":
#             return OpenAIChat()
#         elif chat_type == "phi3-mini-llama":
#             return Phi3MiniChatLlama()
#         elif chat_type == "phi3-mini-hf":
#             return Phi3MiniChatHF()
#         elif chat_type == "tiny-llama":
#             return TinyLlamaChat()
#         else:
#             raise ValueError(f"Invalid chat provider: {chat_type}")


class ChatFactory:
    _registry = {}

    @classmethod
    def register(cls, chat_type):
        def decorator(chat_cls):
            cls._registry[chat_type] = chat_cls
            return chat_cls

        return decorator

    @classmethod
    def get_chat(cls, chat_type):
        if chat_type not in cls._registry:
            raise ValueError(f"Invalid chat provider: {chat_type}")
        return cls._registry[chat_type]()

    @classmethod
    def available_chats(cls):
        return list(cls._registry.keys())
