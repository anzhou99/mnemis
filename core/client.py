import anthropic
from .config import settings
from .models import Message, LLMResponse
from utils.logger import get_logger

import json
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

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

    def structured_chat(
        self,
        messages: list[Message],
        response_model: Type[T],
        system: str | None = None,
        max_retries: int = 3,
    ) -> T:
        """
        让 LLM 输出结构化数据，自动解析并用 Pydantic 验证。
        失败时自动重试，最多 max_retries 次。
        """

        # 把 Pydantic 模型的字段信息注入到 system prompt
        schema_desc = self.build_schema_prompt(response_model)
        full_system = f"{system}\n\n{schema_desc}" if system else schema_desc

        last_error = None
        for attempt in range(1, max_retries + 1):
            logger.debug(f"structured_chat attempt {attempt}/{max_retries}")

            # 如果是重试，把上次的错误反馈给 LLM
            retry_messages = messages.copy()
            if last_error and attempt > 1:
                retry_messages.append(
                    Message(
                        role="user",
                        content=f"你上次的输出解析失败了，错误是：{last_error}\n请重新输出，确保是合法 JSON。",
                    )
                )
            print(f"===输入System===\n {full_system}")
            print(f"===输入Message===\n {retry_messages}")
            response = self.chat(
                messages=retry_messages,
                system=full_system,
                temperature=0.1,  # 结构化输出用低temperature以保持准确性
            )
            print(f"===原始输出===\n {response}")

            try:
                # 清理 LLM 可能附加的 markdown 代码块
                raw = response.content.strip()
                raw = (
                    raw.removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )

                data = json.loads(raw)
                result = response_model(**data)
                logger.debug(f"structured_chat success on attempt {attempt}")
                return result

            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt} failed: {e}")

        raise ValueError(
            f"structured_chat 在 {max_retries} 次尝试后仍然失败。最后错误：{last_error}"
        )

    @staticmethod
    def build_schema_prompt(model: Type[BaseModel]) -> str:
        """从 Pydantic 模型自动生成 JSON 格式说明"""
        return f"""你必须严格按照以下 JSON Schmea指定的格式输出，不要输出任何其他文字、解释或 markdown 标记：
                {{ 
                {model.model_json_schema()}
                }}"""
