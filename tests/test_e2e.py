"""E2E:合成数据 → 证据包 → 直方图评估器 → 落库 → 单调性报告。全程无网络、秒级。"""

import numpy as np
import pytest

from judgelab.core.evidence import EVIDENCE_PACK_VERSION, EvidenceFailure, build_evidence
from judgelab.core.judges import HistogramJudge
from judgelab.core.runner import BenchRunner, load_manifest
from judgelab.core.schema import JudgeInput, JudgeVerdict
from judgelab.domains.color.synth import apply_look, gen_dataset, gen_test_image, random_look


def test_apply_look_identity_and_determinism():
    rng = np.random.default_rng(7)
    img = gen_test_image(rng)
    look = random_look(rng)
    assert np.array_equal(apply_look(img, look, 0.0), img)  # t=0 位级恒等
    a = apply_look(img, look, 0.7)
    b = apply_look(img, look, 0.7)
    assert np.array_equal(a, b)  # 确定性(红线7)
    assert not np.array_equal(a, img)


def test_verdict_schema_rejects_silent_failure():
    with pytest.raises(ValueError):  # 失败还带分 = 静默降级,必须被 schema 拦下(红线1)
        JudgeVerdict(
            case_id="c", judge_id="j", evidence_pack_version="ev1",
            status="judge_input_failure", failure_reason="x",
            dims={"tone_following": {"score": 0.5}},
        )
    with pytest.raises(ValueError):  # ok 却没有维度分,同样非法
        JudgeVerdict(case_id="c", judge_id="j", evidence_pack_version="ev1", status="ok")


def test_evidence_missing_required_raises(tmp_path):
    inp = JudgeInput(case_id="c", source=None, reference=None, result=None)
    with pytest.raises(EvidenceFailure):
        build_evidence(inp, required=("source", "reference", "result"))


def test_e2e_histogram_judge_selfref_vs_real(tmp_path):
    manifest, manifest_selfref = gen_dataset(tmp_path / "data", n_images=4, n_looks=3, seed=0)
    cases = load_manifest(manifest)
    cases_selfref = load_manifest(manifest_selfref)
    assert len(cases) == len(cases_selfref) == 4 * 3 * 5

    runner = BenchRunner(tmp_path / "verdicts.db")
    judge = HistogramJudge()
    stats = runner.run(cases + cases_selfref, judge)
    assert stats["judged"] == len(cases) * 2
    assert stats["statuses"] == {"ok": len(cases) * 2}

    # 缓存生效:重跑零判卷(红线4)
    stats2 = runner.run(cases, judge)
    assert stats2["judged"] == 0 and stats2["skipped_cached"] == len(cases)

    # 自参考设定(reference=同源伪 GT,CanonCGT 式回避协议):直方图评估器组内必须单调 —— 阳性对照
    def _grouped(case_ids: set[str]) -> float:
        import json as _json

        rows = runner.db.execute(
            "SELECT verdict_json, meta_json FROM verdicts WHERE judge_id=? AND status='ok'", (judge.judge_id,)
        ).fetchall()
        from judgelab.core.runner import _spearman

        groups: dict[str, list[tuple[float, float]]] = {}
        for vj, mj in rows:
            v, m = _json.loads(vj), _json.loads(mj)
            if v["case_id"] not in case_ids:
                continue
            groups.setdefault(m["group"], []).append((m["gt_strength"], v["dims"]["tone_following"]["score"]))
        rhos = [r for pairs in groups.values() if (r := _spearman([p[0] for p in pairs], [p[1] for p in pairs])) is not None]
        return sum(rhos) / len(rhos)

    rho_selfref = _grouped({c.case_id for c in cases_selfref})
    rho_real = _grouped({c.case_id for c in cases})
    assert rho_selfref > 0.9, rho_selfref
    # 真实设定(内容不同):同一评估器同一批 result,单调性崩塌 —— 论文核心论点的最小可运行证据
    assert rho_real < rho_selfref - 0.3, (rho_real, rho_selfref)

    # 证据版本铁律字段在库里
    row = runner.db.execute("SELECT DISTINCT evidence_pack_version FROM verdicts").fetchall()
    assert row == [(EVIDENCE_PACK_VERSION,)]
