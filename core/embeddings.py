# core/embeddings.py
import asyncio
from typing import Literal, overload
import numpy as np
from volcenginesdkarkruntime import Ark, AsyncArk
from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局客户端（单例）
_embedding_client: Ark | None = None


@overload
def get_embedding_client(use_async: Literal[True]) -> AsyncArk: ...
@overload
def get_embedding_client(use_async: Literal[False] = False) -> Ark: ...


def get_embedding_client(use_async: bool = False):
    global _embedding_client

    if use_async:
        return AsyncArk(
            api_key=settings.anthropic_api_key,
            base_url=settings.embedding_model_base_url,
            max_retries=3,
        )

    if _embedding_client is None:
        _embedding_client = Ark(
            api_key=settings.anthropic_api_key,
            base_url=settings.embedding_model_base_url,
        )
    return _embedding_client


class Embeddings:

    def __init__(self) -> None:
        pass

    def embed_text(self, text: str) -> list[float]:
        """
        把单段文本转换为向量。
        注意：text 不能为空字符串，API 会报错。
        """
        text = text.strip()
        if not text:
            raise ValueError("Cannot embed empty text")

        client = get_embedding_client()
        response = client.multimodal_embeddings.create(
            model=settings.embedding_model,
            input=[{"type": "text", "text": text}],
            dimensions=settings.embedding_dim,
        )

        vector = response.data.embedding
        logger.debug(f"Embedded text ({len(text)} chars) → vector dim={len(vector)}")
        return vector

    async def _embed_batch_core(self, items_list: list[str]) -> list[list[float]]:
        """核心：异步批量处理向量任务（全组失败）"""
        # 1. 初始化批量客户端
        client = get_embedding_client(use_async=True)
        print(f"****批量处理开始，任务总计 {len(items_list)} 组")
        async with client:
            # 2. 批量创建异步任务（按输入列表分片）
            batch_tasks = [
                asyncio.create_task(
                    client.multimodal_embeddings.create(
                        timeout=30,  # 增加超时时间到30秒
                        model=settings.embedding_model,
                        input=[{"type": "text", "text": text}],
                        dimensions=settings.embedding_dim,
                    )
                )
                for text in items_list
            ]

            try:
                # 3. 批量等待任务完成（任一失败则全组终止）
                batch_results = await asyncio.gather(*batch_tasks)

                print(f"****✅批量处理完成")
                return [b.data.embedding for b in batch_results]
            except Exception as e:
                # 4. 批量清理未完成任务
                for task in batch_tasks:
                    if not task.done():
                        task.cancel()
                print(f"****❌批量处理失败")
                print(e)
                raise

    def embed_batch(self, items_list: list[str]) -> list[list[float]]:
        return asyncio.run(self._embed_batch_core(items_list))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    计算两个向量的余弦相似度。
    返回值：-1 到 1，越接近 1 越相似。
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)
