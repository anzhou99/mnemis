"""
工具包入口。
import 这个包会自动触发所有工具模块的加载，
从而执行 @tool 装饰器，把工具注册进全局注册表。
"""

# 按顺序 import，触发装饰器注册
from core.tools import search      
from core.tools import file_ops   
from core.tools import rag_tools

# 未来新增工具：在这里加一行 import 就够了
# from core.tools import code_exec  