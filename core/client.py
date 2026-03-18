import anthropic
from .config import settings
from .models import Message, LLMResponse
from utils.logger import get_logger


logger = get_logger(__name__)


class LLMClient:
    def __init__(self):
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, base_url=settings.anthropic_base_url
        )

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else settings.temperature

        logger.debug(
            f"LLM call | messages={len(messages)} | temp={temp} | stream={stream}"
        )

        kwargs = {
            "model": settings.default_model,
            "max_tokens": settings.max_tokens,
            "temperature": temp,
            "messages": [m.model_dump() for m in messages],
        }

        if system:
            kwargs["system"] = system

        if stream:
            return self._stream_chat(kwargs)

        response = self._client.messages.create(**kwargs)

        result = LLMResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )

        print(result)

        logger.debug(
            f"Done | in_tokens={result.input_tokens} | out_tokens={result.output_tokens}"
            f" | cost={result.cost_estimate:.4f}"
        )
        return result

    def _stream_chat(self, kwargs: dict) -> LLMResponse:
        full_text = ""
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_text += text
            print()
            final = stream.get_final_message()

        result = LLMResponse(
            content=full_text,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            model=final.model,
        )

        logger.debug(
            f"Stream Done | in_tokens={result.input_tokens} | out_tokens={result.output_tokens}"
            f" | cost={result.cost_estimate:.4f}"
        )

        return result
