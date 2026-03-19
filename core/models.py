from pydantic import BaseModel
from typing import Literal

input_tokens_price = 2 / 100_0000
output_tokens_price = 3 / 100_0000


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_estimate(self) -> float:
        return (
            self.input_tokens * input_tokens_price
            + self.output_tokens * output_tokens_price
        )
