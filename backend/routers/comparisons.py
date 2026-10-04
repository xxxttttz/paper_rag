"""Technical comparison endpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import comparison_service
import database
from backend.schemas import TextCitationResponse
from comparison_models import ComparisonReport, ComparisonRequest
from retriever import ComparisonIndexError


router = APIRouter(prefix="/comparisons", tags=["comparisons"])


class ComparisonResponse(BaseModel):
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    answer: str
    report: ComparisonReport
    citations: list[TextCitationResponse]


@router.post("", response_model=ComparisonResponse)
def compare_solutions(payload: ComparisonRequest) -> dict:
    if database.get_conversation(payload.conversation_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    try:
        return comparison_service.compare(payload)
    except ComparisonIndexError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="方案对比生成失败，请检查检索与模型服务后重试。") from error
