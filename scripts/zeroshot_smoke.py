"""M2 小样本冒烟:VLM zero-shot 判官 × 合成追色集,硬闸 ≤20 条(红线 3)。

用法(出口自选,填对应 env):
  JUDGE_API_BASE=http://127.0.0.1:3001/v1 JUDGE_API_MODEL=<vision模型> JUDGE_API_KEY=<key> \
      uv run python scripts/zeroshot_smoke.py

跑完看三件事:解析成功率(judge_parse_failure 占比)、组内 Spearman、单条耗时/开销。
达标(解析 ≥80%、rho 明显高于直方图 baseline 的 0.58)才谈放量。
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from judgelab.core.judges import ApiJudge, HistogramJudge
from judgelab.core.runner import BenchRunner, load_manifest
from judgelab.domains.color.synth import gen_dataset

SMOKE_LIMIT = 20  # 硬闸:放量前不许改

PROMPT = """You are a color-grading judge. You will see three images with evidence ids:
- source (img:source): the original photo
- reference (img:reference): the target look, applied to DIFFERENT content
- result (img:result): the source after a grading attempt

Judge how well the result follows the reference's color style (tone_following) while
preserving the source's content (content_fidelity). The reference shows different content —
judge the COLOR STYLE only, never content similarity.

Respond with ONLY a JSON object:
{"tone_following": {"score": <0..1>, "evidence_refs": ["img:result", "img:reference"], "rationale": "<one sentence citing concrete color properties>"},
 "content_fidelity": {"score": <0..1>, "evidence_refs": ["img:result", "img:source"], "rationale": "<one sentence>"}}
Scores: 1.0 = perfect. Cite only evidence ids you were actually given."""


def main() -> None:
    if "JUDGE_API_MODEL" not in os.environ:
        sys.exit("set JUDGE_API_BASE / JUDGE_API_MODEL / JUDGE_API_KEY first (pick an outlet)")

    work = Path(tempfile.mkdtemp(prefix="judgelab-smoke-"))
    manifest, _ = gen_dataset(work / "data", n_images=2, n_looks=2, seed=0)  # 20 cases
    cases = load_manifest(manifest)[:SMOKE_LIMIT]
    runner = BenchRunner(work / "verdicts.db")

    hist = HistogramJudge()
    runner.run(cases, hist)
    base = runner.report_monotonic(hist.judge_id, "tone_following", group_key="group")

    api = ApiJudge(prompt=PROMPT, dims=("tone_following", "content_fidelity"))
    t0 = time.time()
    stats = runner.run(cases, api)
    dt = time.time() - t0
    rep = runner.report_monotonic(api.judge_id, "tone_following", group_key="group")

    print(f"cases={len(cases)}  api={api.judge_id}")
    print(f"statuses={stats['statuses']}  wall={dt:.1f}s  ({dt / max(1, stats['judged']):.1f}s/case)")
    print(f"grouped spearman: api={rep}  hist_baseline={base}")
    ok = stats["statuses"].get("ok", 0)
    print("VERDICT:", "PASS — 可以谈放量" if ok / len(cases) >= 0.8 else "FAIL — 先修 prompt/解析,不放量")
    print(f"artifacts: {work}")


if __name__ == "__main__":
    main()
