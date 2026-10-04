"""Contracts for evidence-backed IoT identification comparisons."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Dimension = Literal["data", "method", "deployment", "evaluation", "limitations"]
DIMENSIONS = {
    "data": "数据要求",
    "method": "识别方法",
    "deployment": "部署条件",
    "evaluation": "实验依据",
    "limitations": "适用范围与局限",
}


class ComparisonRequest(BaseModel):
    conversation_id: str = Field(min_length=1)
    sources: list[str] = Field(min_length=2, max_length=3)
    requirements: str = Field(min_length=1, max_length=4000)

    @field_validator("sources")
    @classmethod
    def distinct_sources(cls, values):
        cleaned = [value.strip() for value in values]
        if any(not value or len(value) > 256 for value in cleaned):
            raise ValueError("论文名称不能为空，且不能超过 256 个字符")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("请选择不同的论文")
        return cleaned

    @field_validator("requirements")
    @classmethod
    def nonblank_requirements(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("请填写部署需求")
        return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Support(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=20)
    quote: str = Field(min_length=1, max_length=1200)


class Cell(StrictModel):
    source: str
    dimension: Dimension
    text: str = Field(min_length=1, max_length=2000)
    evidence: list[Support] = Field(default_factory=list, max_length=5)


class Assessment(StrictModel):
    source: str
    text: str = Field(min_length=1, max_length=2000)
    evidence: list[Support] = Field(default_factory=list, max_length=5)


class Comparability(StrictModel):
    status: Literal["unknown", "not_comparable", "comparable"] = "unknown"
    explanation: str = Field(default="实验条件与指标证据不足，无法直接比较效果或判断优劣。", max_length=2000)
    evidence: list[Support] = Field(default_factory=list, max_length=10)


class ComparisonDraft(StrictModel):
    cells: list[Cell] = Field(default_factory=list, max_length=15)
    assessments: list[Assessment] = Field(default_factory=list, max_length=3)
    comparability: Comparability = Field(default_factory=Comparability)


class ComparisonReport(ComparisonDraft):
    sources: list[str]
    requirements: str
