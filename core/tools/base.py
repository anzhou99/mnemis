from pydantic import BaseModel


class ToolSchema(BaseModel):
    """单个工具的完整Schema，直接对应tools格式"""

    name: str
    description: str
    input_schema: dict


class ToolCall(BaseModel):
    """模型返回的 tool_use block 解析后的结构"""

    id: str
    name: str
    input: dict


class ToolResult(BaseModel):
    """工具执行结果，准备回传给模型"""

    tool_use_id: str
    content: str
    is_error: bool = False

    def to_api_format(self) -> dict:
        """转换为 API 期望的 tool_result 格式"""
        result = {
            "types": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            result["is_error"] = True
        return result
