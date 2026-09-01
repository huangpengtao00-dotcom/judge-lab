"""判官 IO 契约。所有判官(API/Mock/本地权重)共用同一输入输出形状。

红线 1:失败必须走显式 status,禁止用空 dims/默认分表达"没判成"。
红线 6:verdict 永远携带 evidence_pack_version,离线重判时按版本对齐。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

VerdictStatus = Literal["ok", "abstain", "judge_input_failure", "judge_parse_failure"]


class JudgeInput(BaseModel):
    """一个判卷 case 的原始输入。三张图允许缺项(领域决定哪些必需)。"""

    case_id: str
    source: Path | None = None
    reference: Path | None = None
    result: Path | None = None
    meta: dict = Field(default_factory=dict)


class DimScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    # 证据引用:必须是证据包里真实存在的 evidence id,runner 会校验
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str = ""


class JudgeVerdict(BaseModel):
    case_id: str
    judge_id: str
    evidence_pack_version: str
    status: VerdictStatus
    # status != "ok" 时 dims 必须为空;"ok" 时至少一维
    dims: dict[str, DimScore] = Field(default_factory=dict)
    failure_reason: str = ""

    @model_validator(mode="after")
    def _status_dims_consistency(self) -> "JudgeVerdict":
        if self.status == "ok" and not self.dims:
            raise ValueError("status=ok requires at least one dimension score")
        if self.status != "ok" and self.dims:
            raise ValueError(f"status={self.status} must not carry dims (no silent partial verdicts)")
        if self.status != "ok" and not self.failure_reason:
            raise ValueError(f"status={self.status} requires failure_reason")
        return self

    def overall(self) -> float | None:
        if self.status != "ok":
            return None
        return sum(d.score for d in self.dims.values()) / len(self.dims)
