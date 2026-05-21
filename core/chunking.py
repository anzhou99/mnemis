import re
import tiktoken
from pathlib import Path
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger(__name__)

# 用 tiktoken 精确计算 token 数（而不是用字符数估算）
_tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI Embedding 用这个编码


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


@dataclass
class Chunk:
    """
    一个文本片段，携带完整的元数据。
    元数据在检索后用于溯源（告诉用户这段内容来自哪里）。
    """

    text: str
    source: str  # 文件名或来源标识
    chunk_index: int  # 在原文档中的位置序号
    token_count: int  # 精确 token 数

    # 可选元数据
    page_num: int | None = None
    section: str | None = None
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Chunk(source={self.source!r}, index={self.chunk_index}, "
            f"tokens={self.token_count}, text={self.text[:50]!r}...)"
        )


class RecursiveChunker:
    """
    递归切割器。
    优先按语义边界（段落→句子→字符）切割，
    保证每个 Chunk 不超过 max_tokens，同时语义尽量完整。
    """

    # 切割优先级：从大到小的语义边界
    SEPARATORS = [
        "\n\n",  # 段落（最优先）
        "\n",  # 换行
        "。",
        "！",
        "？",  # 中文句子
        ". ",
        "! ",
        "? ",  # 英文句子
        "；",
        "; ",  # 分号
        "，",
        ", ",  # 逗号（最后手段）
        " ",  # 空格
        "",  # 字符级（最后防线）
    ]

    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 60) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_text(self, text: str, source: str) -> list[Chunk]:
        """把一段文本切割成 Chunk 列表"""
        text = self._clean_text(text)
        if not text:
            return []

        raw_chunks = self._split_recursive(text, self.SEPARATORS)
        chunks = self._merge_and_overlap(raw_chunks, source)

        logger.debug(
            f"Chunked '{source}': {len(chunks)} chunks "
            f"(avg {sum(c.token_count for c in chunks) // max(len(chunks), 1)} tokens)"
        )
        return chunks

    def _clean_text(self, text: str) -> str:
        """基础清理：去除多余空行和空白"""
        text = re.sub(r"\n{3,}", "\n\n", text)  # 超过两个连续换行 → 两个
        text = re.sub(r"[ \t]+", " ", text)  # 多个空格 → 一个
        return text.strip()

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """
        核心递归逻辑：
        1. 用当前 separator 切割
        2. 对超出大小的块，用下一级 separator 继续切
        3. 直到所有块都满足大小要求，或没有更多 separator
        """
        if not separators:
            # 最后手段：强制按字符数切
            return self._force_split(text)

        sep = separators[0]
        parts = text.split(sep) if sep else list(text)

        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if count_tokens(part) <= self.max_tokens:
                result.append(part)
            else:
                # 这个部分还太大，用更细粒度的 separator 继续切
                sub_chunks = self._split_recursive(part, separators[1:])
                result.extend(sub_chunks)

        return result

    def _force_split(self, text: str) -> list[str]:
        """当所有语义边界都用完时，强制按 token 数切割"""
        tokens = _tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunks.append(_tokenizer.decode(chunk_tokens))
        return chunks

    def _merge_and_overlap(self, parts: list[str], source: str) -> list[Chunk]:
        """
        把切割好的小块合并到接近 max_tokens，
        然后在相邻块之间加入 overlap。
        """
        # 先合并太小的块
        merged = []
        current = ""
        for part in parts:
            candidate = (current + "\n\n" + part).strip() if current else part
            if count_tokens(candidate) <= self.max_tokens:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = part
        if current:
            merged.append(current)

        # 加入 overlap：每个块前面附上上一块的结尾
        chunks = []
        for i, text in enumerate(merged):
            if i > 0 and self.overlap_tokens > 0:
                # 取上一块末尾的 overlap_tokens 个 token
                prev_tokens = _tokenizer.encode(merged[i - 1])
                overlap_text = _tokenizer.decode(prev_tokens[-self.overlap_tokens :])
                text = overlap_text.strip() + "\n" + text

            chunks.append(Chunk(text, source, 1, count_tokens(text)))

        return chunks


class DocumentLoader:
    """从不同格式的文件里提取纯文本"""

    @staticmethod
    def load(file_path: str | Path) -> list[dict]:
        """
        返回 list[{"text": str, "page": int | None}]
        PDF 按页返回，其他格式返回单条。
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return DocumentLoader._load_pdf(path)    
        elif suffix in (".md"):
            return DocumentLoader._load_markdown_with_headers(path)
        elif suffix in (".txt"):
            return DocumentLoader._load_text(path)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. Supported: pdf, md, txt, rst"
            )

    @staticmethod
    def _load_pdf(path: Path) -> list[dict]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():  # 跳过空白页
                pages.append({"text": text, "page": i})
        logger.debug(f"Loaded PDF: {path.name} ({len(pages)} non-empty pages)")
        return pages

    @staticmethod
    def _load_text(path: Path) -> list[dict]:
        text = path.read_text(encoding="utf-8")
        return [{"text": text, "page": None}]
        
    @staticmethod
    def _load_markdown_with_headers(path: Path) -> list[dict]:
        """
        Markdown 文件按一级标题（#）切分成独立段落，
        每段保留标题信息作为元数据。
        比直接整体切割的效果更好。
        """
        text = path.read_text(encoding="utf-8")
        sections = re.split(r'\n(?=# )', text)

        result = []
        for section in sections:
            if not section.strip():
                continue
            # 提取标题
            lines = section.strip().split('\n')
            title = lines[0].lstrip('#').strip() if lines[0].startswith('#') else None
            result.append({"text": section, "page": None, "section": title})

        return result


def chunk_document(
    file_path: str | Path, max_tokens: int = 400, overlap_tokens: int = 60
) -> list[Chunk]:
    """
    完整文档处理管道的统一入口：
    文件 → 文本提取 → 递归切割 → 带元数据的 Chunk 列表
    """
    path = Path(file_path)
    chunker = RecursiveChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    pages = DocumentLoader.load(path)

    all_chunks = []
    global_index = 0

    for page_data in pages:
        page_chunks = chunker.chunk_text(page_data["text"], source=path.name)
        for chunk in page_chunks:
            chunk.chunk_index = global_index
            chunk.page_num = page_data["page"]
            global_index += 1

        all_chunks.extend(page_chunks)

    logger.info(
        f"Document processed: {path.name} → "
        f"{len(all_chunks)} chunks, "
        f"{sum(c.token_count for c in all_chunks)} total tokens"
    )

    return all_chunks
