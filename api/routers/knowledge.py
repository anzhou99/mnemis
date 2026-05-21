# api/routers/knowledge.py
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from api.dependencies import get_current_user

router = APIRouter(prefix="/knowledge", tags=["知识库"])


class AddTextRequest(BaseModel):
    content: str
    source_name: str


@router.post("/add-text")
def add_text(
    body: AddTextRequest,
    current_user: dict = Depends(get_current_user),
):
    """直接添加文本到知识库"""
    from core.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    count = kb.add_text(body.content, source=body.source_name)
    return {"message": f"已添加 {count} 个片段", "source": body.source_name}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """上传文档文件（PDF / Markdown / TXT）"""
    import tempfile
    from pathlib import Path
    from core.knowledge_base import KnowledgeBase

    # 保存到临时文件
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".md", ".txt"):
        raise HTTPException(status_code=400, detail="只支持 PDF、Markdown、TXT 文件")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        kb = KnowledgeBase()
        count = kb.add_document(tmp_path)
        return {
            "message": f"文档已处理",
            "filename": file.filename,
            "chunks": count,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/sources")
def list_sources(current_user: dict = Depends(get_current_user)):
    """列出知识库中所有文档来源"""
    from core.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    return {
        "sources": kb.list_sources(),
        "stats": kb.get_stats(),
    }


@router.delete("/sources/{source_name}")
def delete_source(
    source_name: str,
    current_user: dict = Depends(get_current_user),
):
    """删除指定来源的所有内容"""
    from core.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.delete_source(source_name)
    return {"message": f"已删除来源：{source_name}"}
