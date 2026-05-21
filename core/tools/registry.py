from typing import Callable, Any
from core.tools.base import ToolSchema, ToolResult
from utils.logger import get_logger


logger = get_logger(__name__)


class ToolRegistry:
    """
    全局工具注册表。
    存储所有已注册工具的 Schema 和对应的执行函数。
    设计为单例——整个程序只有一个注册表实例。
    """

    tools: dict[str, dict]

    def __init__(self):
        # name → {"schema": ToolSchema, "fn": Callable}
        self._tools = {}

    def register(self, schema: ToolSchema, fn: Callable):
        """注册一个工具"""
        if schema.name in self._tools:
            logger.warning(f"Tool '{schema.name}' already registered, overwriting.")
        self._tools[schema.name] = {"schema": schema, "fn": fn}
        logger.debug("Registered tool: " + schema.name)

    def get_schemas(self, names: list[str] | None = None) -> list[ToolSchema]:
        """
        获取工具 Schema 列表，用于传给 LLM。
        names 为 None 时返回所有工具；否则只返回指定名称的工具子集。
        """
        if names is None:
            return [entry["schema"] for entry in self._tools.values()]

        return [self._tools[name]["schema"] for name in names if name in self._tools]

    def execute(self, name: str, tool_input: dict) -> ToolResult:
        """
        执行指定工具，返回 ToolResult。
        工具不存在或执行报错时，返回 is_error=True 的 ToolResult，
        不抛出异常——让 Agent 自己决定如何处理错误。
        """
        if name not in self._tools:
            logger.error(f"Unkown tool: {name}")
            return ToolResult(
                tool_use_id="",
                content=f"工具 '{name}' 不存在。可用工具：{self.list_names()}",
                is_error=True,
            )

        fn = self._tools[name]["fn"]
        logger.debug(f"Executing tool: {name} | input={tool_input}")

        try:
            result = fn(**tool_input)
            # 工具函数返回 ToolResult 或字符串，统一处理
            if isinstance(result, ToolResult):
                return result
            return ToolResult(tool_use_id="", content=str(result))

        except TypeError as e:
            # 参数不匹配（模型传了错误的参数名或缺少必填参数）
            logger.error(f"Tool '{name}' parameter error: {e}")
            return ToolResult(
                tool_use_id="",
                content=f"工具参数错误：{e}",
                is_error=True,
            )

        except Exception as e:
            logger.error(f"Tool '{name}' execution failed: {e}")
            return ToolResult(
                tool_use_id="",
                content=f"工具执行失败：{e}",
                is_error=True,
            )

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegister({self.list_names()})"


_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """获取全局工具注册表"""
    return _registry


def tool(schema: ToolSchema):
    """
    工具注册装饰器。
    用法：
        @tool(schema=MY_SCHEMA)
        def my_tool_function(param: str) -> str:
            ...

    装饰后的函数行为不变，只是同时注册进全局注册表。
    """

    def decorator(fn: Callable) -> Callable:
        _registry.register(schema=schema, fn=fn)
        return fn

    return decorator
