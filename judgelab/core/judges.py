"""判官实现:统一 Judge 接口。

- HistogramJudge:传统指标判官(Lab 直方图相似度)。它既是 E2E 冒烟的零成本判官,
  也是论文里"传统指标 baseline"的正式实现之一 —— 它在"内容不同"的设定下会犯的错,
  正是我们要系统展示的。
- ApiJudge:VLM 判官,走 OpenAI 兼容网关。红线 3:放量前先小样本。
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

import numpy as np

from .evidence import EvidencePack
from .schema import DimScore, JudgeVerdict


class Judge(Protocol):
    judge_id: str

    def judge(self, pack: EvidencePack) -> JudgeVerdict: ...


def _hist_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """1 - 平均 L1/2 距离,∈[0,1]。经典直方图口径,故意不做任何内容感知。"""
    return float(1.0 - 0.5 * np.abs(h1 - h2).sum(axis=1).mean())


class HistogramJudge:
    """tone_following = result 与 reference 的 Lab 直方图相似度(内容不同时会把内容差当色调差 —— 特性,非 bug)。
    content_fidelity = result 与 source 的亮度直方图相似度。"""

    judge_id = "hist-lab-v1"

    def judge(self, pack: EvidencePack) -> JudgeVerdict:
        need = {"hist:reference", "hist:result", "hist:source"}
        missing = need - pack.ids()
        if missing:
            return JudgeVerdict(
                case_id=pack.case_id,
                judge_id=self.judge_id,
                evidence_pack_version=pack.version,
                status="judge_input_failure",
                failure_reason=f"missing evidence: {sorted(missing)}",
            )
        h_ref = pack.items["hist:reference"].payload
        h_res = pack.items["hist:result"].payload
        h_src = pack.items["hist:source"].payload
        tone = _hist_similarity(h_res[1:], h_ref[1:])  # 只看 a/b 色度通道
        fidelity = _hist_similarity(h_res[:1], h_src[:1])  # 只看 L 通道
        return JudgeVerdict(
            case_id=pack.case_id,
            judge_id=self.judge_id,
            evidence_pack_version=pack.version,
            status="ok",
            dims={
                "tone_following": DimScore(
                    score=max(0.0, min(1.0, tone)),
                    evidence_refs=["hist:result", "hist:reference"],
                    rationale="Lab a/b histogram similarity (content-blind by design)",
                ),
                "content_fidelity": DimScore(
                    score=max(0.0, min(1.0, fidelity)),
                    evidence_refs=["hist:result", "hist:source"],
                    rationale="L histogram similarity to source",
                ),
            },
        )


class ApiJudge:
    """VLM 判官。base_url/model 从 env 取(JUDGE_API_BASE / JUDGE_API_MODEL / JUDGE_API_KEY)。
    解析失败 → judge_parse_failure;evidence_refs 引用不存在的证据 id → 同样判失败(防编造理由)。"""

    def __init__(self, prompt: str, dims: tuple[str, ...], judge_id: str | None = None):
        self.base_url = os.environ.get("JUDGE_API_BASE", "http://127.0.0.1:3001/v1")
        self.model = os.environ.get("JUDGE_API_MODEL", "batch-cheap")
        self.api_key = os.environ.get("JUDGE_API_KEY", "sk-local")
        self.prompt = prompt
        self.dims = dims
        self.judge_id = judge_id or f"api:{self.model}"

    def judge(self, pack: EvidencePack) -> JudgeVerdict:
        import httpx

        content: list[dict] = [{"type": "text", "text": self.prompt}]
        for role in ("source", "reference", "result"):
            key = f"img:{role}"
            if key in pack.items:
                content.append({"type": "text", "text": f"[{role} image, evidence id={key}]"})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{pack.items[key].payload}"},
                    }
                )
            stats_key = f"lab_stats:{role}"
            if stats_key in pack.items:
                content.append(
                    {"type": "text", "text": f"[evidence id={stats_key}] {json.dumps(pack.items[stats_key].payload)}"}
                )
        try:
            # trust_env=False:绕过本机代理(HTTP(S)_PROXY 会劫持内网/网关请求,老坑)
            with httpx.Client(trust_env=False, timeout=120) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": content}],
                        "temperature": 0.0,
                    },
                )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return self._fail(pack, "judge_input_failure", f"api error: {e}")

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return self._fail(pack, "judge_parse_failure", f"no JSON in output: {text[:200]}")
        try:
            data = json.loads(m.group(0))
            dims = {
                k: DimScore(
                    score=float(v["score"]),
                    evidence_refs=list(v.get("evidence_refs", [])),
                    rationale=str(v.get("rationale", "")),
                )
                for k, v in data.items()
                if k in self.dims
            }
        except Exception as e:
            return self._fail(pack, "judge_parse_failure", f"bad schema: {e}; raw: {text[:200]}")
        if not dims:
            return self._fail(pack, "judge_parse_failure", f"none of expected dims present: {list(data)}")
        # 证据引用必须真实存在(防编造)
        for name, d in dims.items():
            fake = set(d.evidence_refs) - pack.ids()
            if fake:
                return self._fail(pack, "judge_parse_failure", f"dim {name} cites nonexistent evidence: {sorted(fake)}")
        return JudgeVerdict(
            case_id=pack.case_id,
            judge_id=self.judge_id,
            evidence_pack_version=pack.version,
            status="ok",
            dims=dims,
        )

    def _fail(self, pack: EvidencePack, status: str, reason: str) -> JudgeVerdict:
        return JudgeVerdict(
            case_id=pack.case_id,
            judge_id=self.judge_id,
            evidence_pack_version=pack.version,
            status=status,  # type: ignore[arg-type]
            failure_reason=reason,
        )
