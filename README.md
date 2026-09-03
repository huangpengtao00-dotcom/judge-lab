# judge-lab

Evidence-grounded visual judges for image agents. 研究仓:可验证合成数据 → 评估器协议 → 基准与证伪 → (M3) rank-GRPO 训练。

## 状态(2026-09-02,M0 完成)

```bash
uv pip install -e ".[test]" && uv run pytest -q   # 4 passed
```

最小闭环已通:参数化 Look 合成"追色程度已知"的三元组(零标注 verifiable 数据)→ 证据包(Lab 直方图/统计/缩略图,版本铁律)→ 评估器(直方图 baseline / VLM API)→ SQLite 落库(内容指纹缓存,离线可重判)→ 单调性体检报告。

**第一个可运行的实验结论**(n=150 cases × 2 settings, seed=0):

| 评估器 | 设定 | 组内 Spearman(评估器分 vs 真实追色强度) |
|---|---|---|
| 直方图(传统指标) | 自参考伪 GT(CanonCGT 式回避协议) | **0.99**(n=150) |
| 直方图(传统指标) | 真实设定(reference 内容 ≠ source) | **0.58**(n=150) |
| zero-shot VLM(Gemini 系视觉模型,经 OpenAI 兼容网关) | 真实设定(同上) | **0.99**(M2 冒烟 n=20,20/20 解析成功,11.9s/case) |

三行合起来就是论文叙事的地基:传统指标靠伪 GT 协议撑着、真实设定下崩塌;zero-shot VLM 已经能吃内容不同的设定;而文献(EditScore/JarvisEvo 消融)证明未训练的 VLM 评估器会被 reward hacking 击穿、理由无证据约束——**训练过的、证据锚定的三元组评估器**就是要补的最后一块。

## 结构

- `judgelab/core/` 领域无关:schema(verdict 契约,禁静默降级)、evidence(证据包+版本铁律)、judges(HistogramJudge / ApiJudge)、runner(落库+缓存+体检报告)
- `judgelab/domains/color/` 追色合成器(Look 参数化,t=0 位级恒等,同 seed 同字节)
- `judgelab/domains/underwater/` 水下退化合成器(通道衰减+背散射雾幕+深度梯度+低照,蓝/绿水两族;同一 verifiable 契约,方向相反:质量分应随 gt_strength 单调降)——**两条候选方向的数据发生器都已就绪**
- 设计红线见 `CLAUDE.md`

## 下一步

- M1:方向拍板 → 接真实数据集(FiveK/PPR10K + 真 LUT 库,或水下 UIEB/URankerSet)
- M2:ApiJudge 多模型 zero-shot 基线(先 ≤20 条小样本,红线 3)
- M3:rank-GRPO(照 VisualQuality-R1 开源实现),进组后用实验室算力
