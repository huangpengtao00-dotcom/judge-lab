# judge-lab

Evidence-grounded visual judges for image agents. 研究仓:可验证合成数据 → 判官协议 → 基准与证伪 → (M3) rank-GRPO 训练。

## 状态(2026-09-02,M0 完成)

```bash
uv pip install -e ".[test]" && uv run pytest -q   # 4 passed
```

最小闭环已通:参数化 Look 合成"追色程度已知"的三元组(零标注 verifiable 数据)→ 证据包(Lab 直方图/统计/缩略图,版本铁律)→ 判官(直方图 baseline / VLM API)→ SQLite 落库(内容指纹缓存,离线可重判)→ 单调性体检报告。

**第一个可运行的实验结论**(n=150 cases × 2 settings, seed=0):

| 设定 | 组内 Spearman(判官分 vs 真实追色强度) |
|---|---|
| 自参考伪 GT(reference=同源满强度,CanonCGT 式回避协议) | **0.99** |
| 真实设定(reference 内容 ≠ source 内容) | **0.58** |

同一个直方图判官、同一批 result——只换 reference 的内容,单调性从近乎完美跌到不可用。这就是"现有指标依赖伪 GT 协议、真实追色场景无自动评测"的最小可运行证据,也是 VLM 判官要打的靶子。

## 结构

- `judgelab/core/` 领域无关:schema(verdict 契约,禁静默降级)、evidence(证据包+版本铁律)、judges(HistogramJudge / ApiJudge)、runner(落库+缓存+体检报告)
- `judgelab/domains/color/` 追色合成器(Look 参数化,t=0 位级恒等,同 seed 同字节)
- `judgelab/domains/underwater/` 占位,周末方向拍板后填
- 设计红线见 `CLAUDE.md`;方案背景见 Obsidian `16-代码架构起步-judge-lab.md`

## 下一步

- M1:方向拍板 → 接真实数据集(FiveK/PPR10K + 真 LUT 库,或水下 UIEB/URankerSet)
- M2:ApiJudge 多模型 zero-shot 基线(先 ≤20 条小样本,红线 3)
- M3:rank-GRPO(照 VisualQuality-R1 开源实现),进组后用实验室算力
