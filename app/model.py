from llama_cpp import Llama


class LLama:
    SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
    SYSTEM_TOKEN = 1788
    USER_TOKEN = 1404
    BOT_TOKEN = 9225
    LINEBREAK_TOKEN = 13

    ROLE_TOKENS = {"user": USER_TOKEN, "bot": BOT_TOKEN, "system": SYSTEM_TOKEN}

    def __init__(self, model_path: str = "../model/gguf-model-q4_1.bin", n_ctx: int = 2_000) -> None:
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_parts=1,
            n_gpu_layers=-1,
        )
        self.tokens = self.get_system_tokens()
        self.model.eval(self.tokens)

    def get_message_tokens(self, role: str, content: str) -> list[int]:
        message_tokens = self.model.tokenize(content.encode("utf-8"))
        message_tokens.insert(1, self.ROLE_TOKENS[role])
        message_tokens.insert(2, self.LINEBREAK_TOKEN)
        message_tokens.append(self.model.token_eos())
        return message_tokens

    def get_system_tokens(self) -> list[int]:
        return self.get_message_tokens("system", self.SYSTEM_PROMPT)

    def interact(
        self,
        user_message: str,
        top_k: int = 30,
        top_p: float = 0.9,
        temperature: float = 0.2,
        repeat_penalty: float = 1.1,
    ) -> str:
        message_tokens = self.get_message_tokens(role="user", content=user_message)
        role_tokens = [self.model.token_bos(), self.BOT_TOKEN, self.LINEBREAK_TOKEN]
        self.tokens += message_tokens + role_tokens
        generator = self.model.generate(
            self.tokens, top_k=top_k, top_p=top_p, temp=temperature, repeat_penalty=repeat_penalty
        )
        token_str = ""
        for token in generator:
            token_str += self.model.detokenize([token]).decode("utf-8", errors="ignore")
            self.tokens.append(token)
            if token == self.model.token_eos():
                break
        return token_str
