"""Bench runner:manifest → 证据包 → 评估器 → verdict 落库(SQLite)→ 报告。

红线 4:证据版本、内容指纹、原始 verdict 全落库;同 (judge, evidence 版本, 内容指纹) 不重跑。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .evidence import EvidenceFailure, build_evidence, content_hash
from .judges import Judge
from .schema import JudgeInput, JudgeVerdict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    case_id TEXT NOT NULL,
    judge_id TEXT NOT NULL,
    evidence_pack_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    verdict_json TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (case_id, judge_id, evidence_pack_version, content_hash)
);
"""


def load_manifest(path: Path) -> list[JudgeInput]:
    cases = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            JudgeInput(
                case_id=row["case_id"],
                source=Path(row["source"]) if row.get("source") else None,
                reference=Path(row["reference"]) if row.get("reference") else None,
                result=Path(row["result"]) if row.get("result") else None,
                meta={k: v for k, v in row.items() if k not in ("case_id", "source", "reference", "result")},
            )
        )
    return cases


class BenchRunner:
    def __init__(self, db_path: Path, required: tuple[str, ...] = ("source", "reference", "result")):
        self.db = sqlite3.connect(db_path)
        self.db.executescript(_SCHEMA)
        self.required = required

    def run(self, cases: list[JudgeInput], judge: Judge, limit: int | None = None) -> dict:
        """跑一批;limit 用于红线 3 的小样本闸门。返回汇总统计。"""
        done = skipped = 0
        statuses: dict[str, int] = {}
        for inp in cases[: limit or len(cases)]:
            chash = content_hash(inp)
            row = self.db.execute(
                "SELECT 1 FROM verdicts WHERE case_id=? AND judge_id=? AND content_hash=?",
                (inp.case_id, judge.judge_id, chash),
            ).fetchone()
            if row:
                skipped += 1
                continue
            try:
                pack = build_evidence(inp, required=self.required)
                verdict = judge.judge(pack)
            except EvidenceFailure as e:
                verdict = JudgeVerdict(
                    case_id=inp.case_id,
                    judge_id=judge.judge_id,
                    evidence_pack_version="ev-failed",
                    status="judge_input_failure",
                    failure_reason=e.reason,
                )
            self.db.execute(
                "INSERT INTO verdicts (case_id, judge_id, evidence_pack_version, content_hash, status, verdict_json, meta_json) VALUES (?,?,?,?,?,?,?)",
                (
                    inp.case_id,
                    judge.judge_id,
                    verdict.evidence_pack_version,
                    chash,
                    verdict.status,
                    verdict.model_dump_json(),
                    json.dumps(inp.meta),
                ),
            )
            self.db.commit()
            statuses[verdict.status] = statuses.get(verdict.status, 0) + 1
            done += 1
        return {"judged": done, "skipped_cached": skipped, "statuses": statuses}

    def report_monotonic(
        self, judge_id: str, dim: str, gt_key: str = "gt_strength", group_key: str | None = None
    ) -> dict:
        """核心报告:评估器的 dim 分与合成 ground truth 的秩相关(Spearman)。

        group_key=None:全局混排相关 —— 内容差异会污染排名,这正是要测量的失效面。
        group_key 给定:按组(如同一 Look×同一源图)内算相关再平均 —— 排除内容混杂后的
        "纯追色强度"敏感度。两个数放在一起,就是评估器的第一张体检表。
        """
        rows = self.db.execute(
            "SELECT verdict_json, meta_json FROM verdicts WHERE judge_id=? AND status='ok'", (judge_id,)
        ).fetchall()
        groups: dict[str, list[tuple[float, float]]] = {}
        for vj, mj in rows:
            v, m = json.loads(vj), json.loads(mj)
            if dim in v["dims"] and gt_key in m:
                g = str(m.get(group_key, "__all__")) if group_key else "__all__"
                groups.setdefault(g, []).append((float(m[gt_key]), float(v["dims"][dim]["score"])))
        rhos = []
        n_total = 0
        for pairs in groups.values():
            n_total += len(pairs)
            if len(pairs) < 3:
                continue
            rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
            if rho is not None:
                rhos.append(rho)
        if not rhos:
            return {"n": n_total, "spearman": None, "note": "not enough ok verdicts"}
        return {"n": n_total, "n_groups": len(rhos), "spearman": round(sum(rhos) / len(rhos), 4)}


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks
