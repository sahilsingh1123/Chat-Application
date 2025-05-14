import os
from llama_cpp import Llama


class DeepSeekChat:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        use_metal: bool = True,
    ):
        """
        Initialize the DeepSeek LLM inference instance.
        :param model_path: Path to the .gguf model file.
        :param n_ctx: Context window size.
        :param n_threads: Number of CPU threads (ignored if Metal GPU is used).
        :param use_metal: Whether to use Metal GPU backend.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            use_metal=use_metal,
        )

    def chat(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] = None,
    ):
        """
        Generate a response to the given prompt.
        :returns: Generated text.
        """
        try:
            response = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["</s>"],
                stream=False,
            )
            return response.get("choices", [{}])[0].get("text", "").strip()
        except Exception as e:
            # Introspect and offer guidance
            raise RuntimeError("Inference failed: " + str(e))


if __name__ == "__main__":
    # Example usage
    mdl = DeepSeekChat(
        "/Users/sahilsingh/coding/github_codes/deepseek-llm-7B-chat-GGUF/deepseek-llm-7b-chat.Q5_K_M.gguf",
        n_ctx=4096,
        n_threads=8,
        use_metal=True,
    )
    user_prompt = "Hello, DeepSeek! How do you work?"
    print(mdl.chat(user_prompt))
